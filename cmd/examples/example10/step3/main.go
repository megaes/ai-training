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
	"encoding/json"
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

// DEFINE A TOOL INTERFACE TO DEFINE WHAT A TOOL NEEDS TO PROVIDE.

// Tool defines the interface that all tools must implement.
type Tool interface {
	Name() string
	ToolDocument() client.D
	Call(ctx context.Context, arguments map[string]any) client.D
}

// =============================================================================

// WE NEED TO ADD TOOL SUPPORT TO THE AGENT AND ADD TWO NEW FIELDS.

// Agent represents the chat agent that can use tools to perform tasks.
type Agent struct {
	client         *client.SSEClient[client.Chat]
	getUserMessage func() (string, bool)

	// ADDING TOOL SUPPORT TO THE AGENT.
	tools         map[string]Tool
	toolDocuments []client.D
}

// NewAgent creates a new instance of Agent.
func NewAgent(sseClient *client.SSEClient[client.Chat], getUserMessage func() (string, bool)) *Agent {

	// PRE-CONSTRUCT ALL THE TOOLS AND TOOL DOCUMENTS.

	gw := NewGetWeather()
	tools := map[string]Tool{
		gw.Name(): gw,
	}

	toolDocs := make([]client.D, 0, len(tools))
	for _, tool := range tools {
		toolDocs = append(toolDocs, tool.ToolDocument())
	}

	// ADD TOOL SUPPORT TO THE AGENT.

	return &Agent{
		client:         sseClient,
		getUserMessage: getUserMessage,
		tools:          tools,
		toolDocuments:  toolDocs,
	}
}

// Run starts the agent and runs the chat loop.
func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D
	var inToolCall bool

	fmt.Printf("Chat with %s (use 'ctrl-c' to quit)\n", model)

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
		}

		inToolCall = false

		d := client.D{
			"model":       model,
			"messages":    conversation,
			"max_tokens":  contextWindow,
			"temperature": 0.0,
			"top_p":       0.1,
			"top_k":       1,
			"stream":      true,

			// ADDING TOOL CALLING TO THE REQUEST.
			"tools":          a.toolDocuments,
			"tool_selection": "auto",
			"options":        client.D{"num_ctx": contextWindow},
		}

		fmt.Printf("\u001b[93m\n%s\u001b[0m: ", model)

		ch := make(chan client.Chat, 100)
		if err := a.client.Do(ctx, http.MethodPost, url, d, ch); err != nil {
			return fmt.Errorf("do: %w", err)
		}

		// WE NEED TO KEEP THE MODEL RESPONSE CHUNKS FOR CONVERSATION HISTORY.
		var chunks []string

		for resp := range ch {
			switch {
			case len(resp.Choices[0].Delta.ToolCalls) > 0:
				results := a.callTools(ctx, resp.Choices[0].Delta.ToolCalls)

				// NOW WE NEED TO CHECK IF THE TOOL CALLS PROVIDED ANY RESULTS
				// TO ADD TO THE CONVERSATION AND MARK WE ARE IN A TOOL CALL.
				if len(results) > 0 {
					conversation = append(conversation, results...)
					inToolCall = true
				}

			case resp.Choices[0].Delta.Content != "":
				fmt.Print(resp.Choices[0].Delta.Content)
				chunks = append(chunks, resp.Choices[0].Delta.Content)

			case resp.Choices[0].Delta.Reasoning != "":
				fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
			}
		}

		// WE NEED TO ADD THE MODEL RESPONSE BACK INTO THE CONVERSATION. WE
		// NEVER ADD THE REASONING BECAUSE IT'S EXTRA TOKENS AND IT WON'T ADD
		// ANYTHING TO THE CONTEXT OF THE CONVERSATION.

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

// WE NEED A FUNCTION THAT LOOKS UP THE REQUESTED TOOL BY NAME AND CALLS IT.

// callTools will lookup a requested tool by name and call it.
func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) []client.D {
	var resps []client.D

	for _, toolCall := range toolCalls {
		tool, exists := a.tools[toolCall.Function.Name]
		if !exists {
			continue
		}

		fmt.Printf("\n\n\u001b[92mtool\u001b[0m: %s(%v)\n", tool.Name(), toolCall.Function.Arguments)

		resp := tool.Call(ctx, toolCall.Function.Arguments)
		resps = append(resps, resp)

		fmt.Printf("%#v\n", resps)
	}

	return resps
}

// =============================================================================

// GetWeather represents a tool that can be used to get the current weather.
type GetWeather struct {
	name string
}

// NewGetWeather creates a new instance of GetWeather.
func NewGetWeather() *GetWeather {
	return &GetWeather{
		name: "get_current_weather",
	}
}

// Name returns the name of the tool.
func (gw *GetWeather) Name() string {
	return gw.name
}

// ToolDocument defines the metadata for the tool that is provied to the model.
func (gw *GetWeather) ToolDocument() client.D {
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
func (gw *GetWeather) Call(ctx context.Context, arguments map[string]any) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = client.D{
				"role":    "tool",
				"name":    gw.name,
				"status":  "FAILED",
				"content": fmt.Sprintf(`{"status": "FAILED", "data": "%s"}`, r),
			}
		}
	}()

	// We are going to hardcode a result for now so we can test the tool.
	// We are going to return the current weather as structured data using JSON
	// which is easier for the model to interpret.

	location := arguments["location"].(string)

	data := map[string]any{
		"temperature": 28,
		"humidity":    80,
		"wind_speed":  10,
		"description": fmt.Sprintln("The weather in", location, "is hot and humid"),
	}

	info := struct {
		Status string         `json:"status"`
		Data   map[string]any `json:"data"`
	}{
		Status: "SUCCESS",
		Data:   data,
	}

	d, err := json.Marshal(info)
	if err != nil {
		return client.D{
			"role":    "tool",
			"name":    gw.name,
			"content": fmt.Sprintf(`{"status": "FAILED", "data": "%s"}`, err),
		}
	}

	return client.D{
		"role":    "tool",
		"name":    gw.name,
		"content": string(d),
	}
}
