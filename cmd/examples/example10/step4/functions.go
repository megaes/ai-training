package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ardanlabs/ai-training/foundation/client"
)

// GetWeather represents a tool that can be used to get the current weather.
type GetWeather struct {
	name string
}

// NewGetWeather creates a new instance of the GetWeather tool and loads it
// into the provided tools map.
func NewGetWeather(tools map[string]Tool) client.D {
	gw := GetWeather{
		name: "tool_get_current_weather",
	}
	tools[gw.name] = &gw

	return gw.toolDocument()
}

// Name returns the name of the tool.
func (gw *GetWeather) Name() string {
	return gw.name
}

// toolDocument defines the metadata for the tool that is provied to the model.
func (gw *GetWeather) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        gw.name,
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

// Call is the function that is called by the agent to get the weather when the
// model requests the tool with the specified parameters.
func (gw *GetWeather) Call(ctx context.Context, toolCall client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(toolCall.ID, gw.name, fmt.Errorf("%s", r))
		}
	}()

	// We are going to hardcode a result for now so we can test the tool.
	// We are going to return the current weather as structured data using JSON
	// which is easier for the model to interpret.

	location := toolCall.Function.Arguments["location"].(string)

	data := map[string]any{
		"temperature": 28,
		"humidity":    80,
		"wind_speed":  10,
		"description": fmt.Sprintln("The weather in", location, "is hot and humid"),
	}

	d, err := json.Marshal(data)
	if err != nil {
		return toolErrorResponse(toolCall.ID, gw.name, err)
	}

	return toolSuccessResponse(toolCall.ID, gw.name, "weather", string(d))
}
