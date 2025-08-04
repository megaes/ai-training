// https://ampcode.com/how-to-build-an-agent
//
// This example shows you how add tool calling to the chat agent from step 1.
//
// # Running the example:
//
//	$ make example10-step3
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
	var conversation []client.D
	var inToolCall bool

	fmt.Println("Chat with qwen3 (use 'ctrl-c' to quit)")

	for {
		// CHECK IF WE ARE IN A TOOL CALL BEFORE ASKING FOR INPUT.
		if !inToolCall {
			fmt.Print("\u001b[94m\nYou\u001b[0m: ")
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}

			conversation = append(conversation, client.D{
				"role":    "user",
				"content": userInput,
			})

			fmt.Print("\u001b[93m\nqwen3\u001b[0m: ")
		}

		inToolCall = false

		d := client.D{
			"model":       "qwen3:8b",
			"messages":    conversation,
			"max_tokens":  1000,
			"temperature": 0.1,
			"top_p":       0.1,
			"top_k":       50,
			"stream":      true,

			// ADDING TOOL CALLING TO THE REQUEST.
			"tools": []client.D{
				GetWeather{}.Tool(),
			},
			"tool_selection": "auto",
			"options":        client.D{"num_ctx": 32000},
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
				switch {
				case len(resp.Message.ToolCalls) > 0:
					// ADD SUPPORT FOR TOOL CALLING.
					result, err := callTools(ctx, resp.Message.ToolCalls)
					if err != nil {
						return fmt.Errorf("call tools: %w", err)
					}

					if len(result) > 0 {
						conversation = append(conversation, result)
						inToolCall = true
					}

				case resp.Message.Content != "":
					fmt.Print(resp.Message.Content)
					chunks = append(chunks, resp.Message.Content)
				}
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

func callTools(ctx context.Context, toolCalls []client.ToolCall) (client.D, error) {
	for _, toolCall := range toolCalls {
		if toolCall.Function.Name == "get_current_weather" {
			var getWeather GetWeather

			resp, err := getWeather.Call(ctx, toolCall.Function.Arguments)
			if err != nil {
				return client.D{}, fmt.Errorf("call: %w", err)
			}

			return resp, nil
		}
	}

	return client.D{}, nil
}

// =============================================================================

type GetWeather struct{}

func (g GetWeather) Tool() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        "get_current_weather",
			"description": "Get the current weather for a location",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"location": client.D{
						"type":        "string",
						"description": "The location to get the weather for, e.g. San Francisco, CA",
					},
				},
				"required": []string{"location"},
			},
		},
	}
}

func (g GetWeather) Call(ctx context.Context, arguments map[string]string) (client.D, error) {
	return client.D{
		"role":    "tool",
		"name":    "get_current_weather",
		"content": "hot and humid, 28 degrees celcius",
	}, nil
}
