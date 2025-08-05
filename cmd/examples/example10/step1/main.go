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

const url = "http://localhost:11434/api/chat"

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

	fmt.Println("Chat with qwen3 (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\u001b[94m\nYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		fmt.Print("\u001b[93m\nqwen3\u001b[0m: ")

		conversation = append(conversation, client.D{
			"role":    "user",
			"content": userInput,
		})

		d := client.D{
			"model":       "qwen3:8b",
			"messages":    conversation,
			"max_tokens":  32768,
			"temperature": 0.1,
			"top_p":       0.1,
			"top_k":       50,
			"stream":      true,
			"options":     client.D{"num_ctx": 32768},
		}

		ch := make(chan client.Chat, 100)
		if err := a.client.Do(ctx, http.MethodPost, url, d, ch); err != nil {
			return fmt.Errorf("do: %w", err)
		}

		var chunks []string

		for resp := range ch {
			if resp.Message.Content != "" {
				fmt.Print(resp.Message.Content)
				chunks = append(chunks, resp.Message.Content)
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
