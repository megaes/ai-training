// https://ampcode.com/how-to-build-an-agent
//
// This example shows you how to create a terminal based chat client.
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

	agent := Agent{
		client:         cln,
		getUserMessage: getUserMessage,
	}

	return agent.Run(context.TODO())
}

// =============================================================================

type Agent struct {
	client         *client.SSEClient[client.Chat]
	getUserMessage func() (string, bool)
}

func (a *Agent) Run(ctx context.Context) error {
	var conversation []string

	fmt.Println("Chat with qwen3 (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\u001b[94m\nYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		fmt.Print("\u001b[93m\nqwen3\u001b[0m: ")

		conversation = append(conversation, userInput)
		conversation = append(conversation, "\n")

		messages := make([]client.D, len(conversation))
		for i, message := range conversation {
			messages[i] = client.D{
				"role":    "user",
				"content": message,
			}
		}

		d := client.D{
			"model":       "qwen3:8b",
			"messages":    messages,
			"max_tokens":  1000,
			"temperature": 0.1,
			"top_p":       0.1,
			"top_k":       50,
			"stream":      true,
		}

		ch := make(chan client.Chat, 100)
		if err := a.client.Do(ctx, http.MethodPost, url, d, ch); err != nil {
			return fmt.Errorf("do: %w", err)
		}

		var chunks []string
		var thinking bool

		for resp := range ch {
			switch resp.Message.Content {
			case "<think>":
				thinking = true
				continue
			case "</think>":
				thinking = false
				continue
			}

			if !thinking {
				fmt.Printf("%s", resp.Message.Content)
				chunks = append(chunks, resp.Message.Content)
			}
		}

		fmt.Print("\n")
		conversation = append(conversation, strings.Join(chunks, " "))
		conversation = append(conversation, "\n")
	}

	return nil
}
