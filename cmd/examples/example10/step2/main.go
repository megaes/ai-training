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
	// if err := nonWeatherQuestion(); err != nil {
	// 	return fmt.Errorf("nonWeatherQuestion: %w", err)
	// }

	// fmt.Println("\n===================================================")

	if err := weatherQuestion(); err != nil {
		return fmt.Errorf("weatherQuestion: %w", err)
	}

	return nil
}

func nonWeatherQuestion() error {
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

	d := client.D{
		"model": "qwen3:8b",
		"messages": []client.D{
			{
				"role":    "user",
				"content": "How do you feel today?",
			},
		},
		"max_tokens":  1000,
		"temperature": 0.1,
		"top_p":       0.1,
		"top_k":       50,
		"stream":      true,
		"tools": []client.D{
			{
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
			},
		},
		"tool_selection": "auto",
		"options":        client.D{"num_ctx": 32000},
	}

	// -------------------------------------------------------------------------

	ch := make(chan client.Chat, 100)
	if err := cln.Do(ctx, http.MethodPost, url, d, ch); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	for resp := range ch {
		fmt.Printf("%s", resp.Message.Content)
		if len(resp.Message.ToolCalls) > 0 {
			fmt.Printf("%s", resp.Message.ToolCalls[0].Function.Name)
			fmt.Printf("%s", resp.Message.ToolCalls[0].Function.Arguments)
		}
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

	d := client.D{
		"model": "qwen3:8b",
		"messages": []client.D{
			{
				"role":    "user",
				"content": "What is the weather like in New York City?",
			},
		},
		"max_tokens":  1000,
		"temperature": 0.1,
		"top_p":       0.1,
		"top_k":       50,
		"stream":      true,
		"tools": []client.D{
			{
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
			},
		},
		"tool_selection": "auto",
		"options":        client.D{"num_ctx": 32000},
	}

	// -------------------------------------------------------------------------

	ch := make(chan client.Chat, 100)
	if err := cln.Do(ctx, http.MethodPost, url, d, ch); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	for resp := range ch {
		fmt.Printf("%s", resp.Message.Content)
		if len(resp.Message.ToolCalls) > 0 {
			fmt.Printf("%s", resp.Message.ToolCalls[0].Function.Name)
			fmt.Printf("%s", resp.Message.ToolCalls[0].Function.Arguments)
		}
	}

	return nil
}

// =============================================================================

type GetWeather struct {
	Location string `json:"location"` // The city and state, e.g. San Francisco, CA
	Format   string `json:"format"`   // The format to return the weather in, e.g. 'celsius' or 'fahrenheit'
}

func (g GetWeather) Name() string {
	return "get_current_weather"
}

func (g GetWeather) Description() string {
	return "Get the current weather for a location"
}

func (g GetWeather) Parameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{
				"type":        "string",
				"description": "The location to get the weather for, e.g. San Francisco, CA",
			},
			"format": map[string]any{
				"type":        "string",
				"description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
				"enum":        []string{"celsius", "fahrenheit"},
			},
		},
		"required": []string{"location", "format"},
	}
}

func (g GetWeather) Call(ctx context.Context, input string) (string, error) {
	return "hot and humid, 28 degrees celcius", nil
}
