// https://ampcode.com/how-to-build-an-agent
//
// This example shows you how to build a simple coding agent.
//
// # Running the example:
//
//	$ make example10
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
	"os"
	"strings"
	"sync"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
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

	llm, err := ollama.New(ollama.WithModel("llama3.2"))
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	agent := Agent{
		client:         llm,
		getUserMessage: getUserMessage,
	}

	return agent.Run(context.TODO())
}

// =============================================================================

type Agent struct {
	client         *ollama.LLM
	getUserMessage func() (string, bool)
}

func (a *Agent) Run(ctx context.Context) error {
	var conversation []string
	var chunks []string

	fmt.Println("Chat with Llama (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\u001b[94m\nYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		fmt.Print("\u001b[93m\nLlama\u001b[0m: ")

		conversation = append(conversation, userInput)
		conversation = append(conversation, "\n")

		var wg sync.WaitGroup
		wg.Add(1)

		f := func(ctx context.Context, chunk []byte) error {
			if ctx.Err() != nil || len(chunk) == 0 {
				conversation = append(conversation, strings.Join(chunks, " "))
				conversation = append(conversation, "\n")
				fmt.Print("\n")
				chunks = []string{}
				wg.Done()
				return nil
			}

			fmt.Printf("%s", chunk)
			chunks = append(chunks, string(chunk))
			return nil
		}

		if _, err := a.client.Call(ctx, strings.Join(conversation, " "), llms.WithStreamingFunc(f)); err != nil {
			return fmt.Errorf("call: %w", err)
		}

		wg.Wait()
	}

	return nil
}
