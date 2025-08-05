// https://ampcode.com/how-to-build-an-agent
//
// This example shows you the workflow and mechanics for tool calling.
//
// # Running the example:
//
//	$ make example10-step2
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
)

const url = "http://localhost:11434/api/chat"

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if err := weatherQuestion(); err != nil {
		return fmt.Errorf("weatherQuestion: %w", err)
	}

	return nil
}

func weatherQuestion() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	logger := func(ctx context.Context, msg string, v ...any) {
		s := fmt.Sprintf("msg: %s", msg)
		for i := 0; i < len(v); i = i + 2 {
			s = s + fmt.Sprintf(", %s: %v", v[i], v[i+1])
		}
		log.Println(s)
	}

	cln := client.NewSSE[client.Chat](logger)

	// -------------------------------------------------------------------------
	// Start by asking what the weather is like in New York City

	var getWeather GetWeather

	q := "What is the weather like in New York City?"
	fmt.Printf("\nQuestion:\n\n%s\n\n", q)

	conversation := []client.D{
		{
			"role":    "user",
			"content": q,
		},
	}

	d := client.D{
		"model":       "qwen3:8b",
		"messages":    conversation,
		"max_tokens":  1000,
		"temperature": 0.1,
		"top_p":       0.1,
		"top_k":       50,
		"stream":      true,
		"tools": []client.D{
			getWeather.ToolDocument(),
		},
		"tool_selection": "auto",
		"options":        client.D{"num_ctx": 32000},
	}

	ch := make(chan client.Chat, 100)
	if err := cln.Do(ctx, http.MethodPost, url, d, ch); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	// -------------------------------------------------------------------------
	// The model will respond asking us to make the get_current_weather function
	// call. We will make the call and then send the response back to the model.

	for resp := range ch {
		fmt.Print(resp.Message.Content)

		if len(resp.Message.ToolCalls) > 0 {
			if resp.Message.ToolCalls[0].Function.Name == "get_current_weather" {
				fmt.Printf("Model Asking For Tool Call:\n\n%s(%s)\n\n", resp.Message.ToolCalls[0].Function.Name, resp.Message.ToolCalls[0].Function.Arguments)

				resp, err := getWeather.Call(ctx, resp.Message.ToolCalls[0].Function.Arguments)
				if err != nil {
					return fmt.Errorf("call: %w", err)
				}

				conversation = append(conversation, resp)

				fmt.Printf("Tool Call Result:\n\n%s\n\n", resp)
			}
		}
	}

	// -------------------------------------------------------------------------
	// Send the result of the tool call back to the model

	d = client.D{
		"model":       "qwen3:8b",
		"messages":    conversation,
		"max_tokens":  32768,
		"temperature": 0.1,
		"top_p":       0.1,
		"top_k":       50,
		"stream":      true,
		"tools": []client.D{
			getWeather.ToolDocument(),
		},
		"tool_selection": "auto",
		"options":        client.D{"num_ctx": 32768},
	}

	ch = make(chan client.Chat, 100)
	if err := cln.Do(ctx, http.MethodPost, url, d, ch); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	// -------------------------------------------------------------------------
	// The model should provide the answer based on the tool call

	fmt.Print("Final Result:\n")

	for resp := range ch {
		fmt.Print(resp.Message.Content)

		if len(resp.Message.ToolCalls) > 0 {
			fmt.Printf("Model Asking For Tool Call:\n\n%s(%s)\n\n", resp.Message.ToolCalls[0].Function.Name, resp.Message.ToolCalls[0].Function.Arguments)
		}
	}

	return nil
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

func (gw GetWeather) Call(ctx context.Context, arguments map[string]any) (client.D, error) {
	return client.D{
		"role":    "tool",
		"name":    gw.Name(),
		"content": "hot and humid, 28 degrees celcius",
	}, nil
}
