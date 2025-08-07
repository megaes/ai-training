// https://ampcode.com/how-to-build-an-agent
//
// This example shows you how to create a terminal based chat agent.
// using the Ollama service and qwen3 model.
//
// # Running the example:
//
//	$ make example10-step1
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
)

const (
	url           = "http://localhost:11434/v1/chat/completions"
	model         = "gpt-oss:latest"
	contextWindow = 168 * 1024 // 168K tokens
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	logger := func(ctx context.Context, msg string, v ...any) {
		s := fmt.Sprintf("msg: %s", msg)
		for i := 0; i < len(v); i = i + 2 {
			s = s + fmt.Sprintf(", %s: %v", v[i], v[i+1])
		}
		log.Println(s)
	}

	cln := client.NewSSE[client.Chat](logger)

	agent := NewAgent(cln, getUserMessage)

	return agent.Run(context.TODO())
}

// =============================================================================

type Agent struct {
	client         *client.SSEClient[client.Chat]
	getUserMessage func() (string, bool)
}

func NewAgent(sseClient *client.SSEClient[client.Chat], getUserMessage func() (string, bool)) *Agent {
	return &Agent{
		client:         sseClient,
		getUserMessage: getUserMessage,
	}
}

func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D

	fmt.Printf("Chat with %s (use 'ctrl-c' to quit)\n", model)

	for {
		fmt.Print("\u001b[94m\nYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		conversation = append(conversation, client.D{
			"role":    "user",
			"content": userInput,
		})

		d := client.D{
			"model":       model,
			"messages":    conversation,
			"max_tokens":  contextWindow,
			"temperature": 0.1,
			"top_p":       0.1,
			"top_k":       50,
			"stream":      true,
			"options":     client.D{"num_ctx": contextWindow},
		}

		fmt.Printf("\u001b[93m\n%s\u001b[0m: ", model)

		ch := make(chan client.Chat, 100)
		if err := a.client.Do(ctx, http.MethodPost, url, d, ch); err != nil {
			return fmt.Errorf("do: %w", err)
		}

		var chunks []string

		thinking := true
		fmt.Print("\n<reasoning>\n")

		for resp := range ch {
			switch {
			case resp.Choices[0].Delta.Content != "":
				if thinking {
					thinking = false
					fmt.Print("\n</reasoning>\n\n")
				}

				fmt.Print(resp.Choices[0].Delta.Content)
				chunks = append(chunks, resp.Choices[0].Delta.Content)

			case resp.Choices[0].Delta.Reasoning != "":
				fmt.Print(resp.Choices[0].Delta.Reasoning)
			}
		}

		if len(chunks) > 0 {
			fmt.Print("\n")

			conversation = append(conversation, client.D{
				"role":    "assistant",
				"content": strings.Join(chunks, " "),
			})
		}
	}

	return nil
}
