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

	agent := NewAgent(cln, getUserMessage)

	return agent.Run(context.TODO())
}

// =============================================================================

// DEFINE A TOOL INTERFACE TO DEFINE WHAT A TOOL NEEDS TO PROVIDE.

type Tool interface {
	Name() string
	ToolDocument() client.D
	Call(ctx context.Context, arguments map[string]string) (client.D, error)
}

// =============================================================================

// WE NEED TO ADD TOOL SUPPORT TO THE AGENT.
// WE WILL ADD TWO NEW FIELDS AND PRE-CONSTRUCT ALL THE TOOLING.

type Agent struct {
	client         *client.SSEClient[client.Chat]
	getUserMessage func() (string, bool)
	tools          []Tool
	toolDocuments  []client.D
}

func NewAgent(sseClient *client.SSEClient[client.Chat], getUserMessage func() (string, bool)) *Agent {
	tools := []Tool{
		NewGetWeather(),
	}

	toolDocs := make([]client.D, len(tools))
	for i, tool := range tools {
		toolDocs[i] = tool.ToolDocument()
	}

	return &Agent{
		client:         sseClient,
		getUserMessage: getUserMessage,
		tools:          tools,
		toolDocuments:  toolDocs,
	}
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
			"tools":          a.toolDocuments,
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
					result, err := a.callTools(ctx, resp.Message.ToolCalls)
					if err != nil {
						fmt.Printf("\n\n\u001b[92m\ntool\u001b[0m: %s", err)
						continue
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

		if !inToolCall && len(chunks) > 0 {
			fmt.Print("\n")

			content := strings.Join(chunks, " ")
			content = strings.TrimLeft(content, "\n")

			if content != "" {
				conversation = append(conversation, client.D{
					"role":    "assistant",
					"content": content,
				})
			}
		}
	}

	return nil
}

func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) (client.D, error) {
	for _, toolCall := range toolCalls {
		for _, tool := range a.tools {
			if toolCall.Function.Name == tool.Name() {
				fmt.Printf("\u001b[92m\ntool\u001b[0m: %s(%s)", tool.Name(), toolCall.Function.Arguments)

				resp, err := tool.Call(ctx, toolCall.Function.Arguments)
				if err != nil {
					return client.D{}, fmt.Errorf("\n\nERROR: %w", err)
				}
				return resp, nil
			}
		}
	}

	return client.D{}, nil
}

// =============================================================================

type GetWeather struct {
	name string
}

func NewGetWeather() GetWeather {
	return GetWeather{
		name: "get_current_weather",
	}
}

func (gw GetWeather) Name() string {
	return gw.name
}

func (gw GetWeather) ToolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        gw.Name(),
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

func (gw GetWeather) Call(ctx context.Context, arguments map[string]string) (client.D, error) {
	return client.D{
		"role":    "tool",
		"name":    gw.Name(),
		"content": "hot and humid, 28 degrees celcius",
	}, nil
}
