// https://ampcode.com/how-to-build-an-agent
//
// This example shows you how add tool calling to the chat agent from
// steps 2 and 3.
//
// # Running the example:
//
//	$ make example10-step4
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
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/tiktoken"
)

const (
	url   = "http://localhost:11434/v1/chat/completions"
	model = "gpt-oss:latest"
)

// The context window represents the maximum number of tokens that can be sent
// and received by the model. The default for Ollama is 8K. In the makefile
// it has been increased to 64K.
var contextWindow = 1024 * 8

func init() {
	if v := os.Getenv("OLLAMA_CONTEXT_LENGTH"); v != "" {
		var err error
		contextWindow, err = strconv.Atoi(v)
		if err != nil {
			log.Fatal(err)
		}
	}
}

// =============================================================================

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

	agent, err := NewAgent(getUserMessage)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	return agent.Run(context.TODO())
}

// =============================================================================

// DECLARE A TOOL INTERFACE TO ALLOW THE AGENT TO CALL ANY TOOL FUNCTION
// WE DEFINE WITHOUT THE AGENT KNOWING THE EXACT TOOL IT IS USING.

// Tool describes the features which all tools must implement.
type Tool interface {
	Call(ctx context.Context, toolCall client.ToolCall) client.D
}

// =============================================================================

// Agent represents the chat agent that can use tools to perform tasks.
type Agent struct {
	sseClient      *client.SSEClient[client.ChatSSE]
	getUserMessage func() (string, bool)
	tke            *tiktoken.Tiktoken

	// WE NEED TO ADD TOOL SUPPORT TO THE AGENT. WE NEED TO HAVE A SET OF
	// TOOLS THAT THE AGENT CAN USE TO PERFORM TASKS AND THE CORRESPONDING
	// DOCUMENTATION FOR THE MODEL.
	tools         map[string]Tool
	toolDocuments []client.D
}

// NewAgent creates a new instance of Agent.
func NewAgent(getUserMessage func() (string, bool)) (*Agent, error) {
	logger := func(ctx context.Context, msg string, v ...any) {
		s := fmt.Sprintf("msg: %s", msg)
		for i := 0; i < len(v); i = i + 2 {
			s = s + fmt.Sprintf(", %s: %v", v[i], v[i+1])
		}
		log.Println(s)
	}

	sseClient := client.NewSSE[client.ChatSSE](logger)

	tke, err := tiktoken.NewTiktoken()
	if err != nil {
		return nil, fmt.Errorf("failed to create tiktoken: %w", err)
	}

	// CONSTRUCT THE TOOLS MAP HERE BECAUSE IT IS PASSED ON TOOL CONSTRUCTION
	// SO TOOLS CAN REGISTER THEMSELVES IN THIS MAP OF AVAILABLE TOOLS.
	tools := map[string]Tool{}

	agent := Agent{
		sseClient:      sseClient,
		getUserMessage: getUserMessage,
		tke:            tke,

		// ADD THE TOOLNG SUPPORT TO THE AGENT.
		tools: tools,
		toolDocuments: []client.D{
			RegisterGetWeather(tools),
		},
	}

	return &agent, nil
}

// WE NEED TO EXTEND THE SYSTEM PROMPT TO INCLUDE THE TOOLING INSTRUCTIONS.

// The system prompt for the model so it behaves as expected.
var systemPrompt = `You are a helpful coding assistant that has tools to assist
you in coding.

After you request a tool call, you will receive a JSON document with two fields,
"status" and "data". Always check the "status" field to know if the call "SUCCEED"
or "FAILED". The information you need to respond will be provided under the "data"
field. If the called "FAILED", just inform the user and don't try using the tool
again for the current response.

When reading Go source code always start counting lines of code from the top of
the source code file.

If you get back results from a tool call, do not verify the results.

Reasoning: high
`

// Run starts the agent and runs the chat loop.
func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D
	var reasonContent []string

	// WE WILL KEEP TRACK OF WHETHER WE ARE IN A TOOL CALL.
	var inToolCall bool

	conversation = append(conversation, client.D{
		"role":    "system",
		"content": systemPrompt,
	})

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

		// WE NEED TO RESET THE TOOL CALL FLAG ON EACH ITERATION.
		inToolCall = false

		d := client.D{
			"model":       model,
			"messages":    conversation,
			"max_tokens":  contextWindow,
			"temperature": 0.1,
			"top_p":       0.1,
			"top_k":       1,
			"stream":      true,

			// ADDING TOOL CALLING TO THE REQUEST.
			"tools":          a.toolDocuments,
			"tool_selection": "auto",
		}

		fmt.Printf("\u001b[93m\n%s\u001b[0m: ", model)

		ch := make(chan client.ChatSSE, 100)
		ctx, cancelContext := context.WithTimeout(ctx, time.Minute*5)

		if err := a.sseClient.Do(ctx, http.MethodPost, url, d, ch); err != nil {
			cancelContext()
			fmt.Printf("\n\n\u001b[91mERROR:%s\u001b[0m\n\n", err)
			continue
		}

		var chunks []string

		reasonThinking := false  // GPT models provide a Reasoning field.
		contentThinking := false // Other reasoning models use <think> tags.
		reasonContent = nil      // Reset the reasoning content for this next call.

		fmt.Print("\n")

		for resp := range ch {
			switch {

			// WE NEED TO CHECK IF WE ARE ASKING TO MAKE A TOOL CALL.
			case len(resp.Choices[0].Delta.ToolCalls) > 0:
				toolCall := resp.Choices[0].Delta.ToolCalls[0]

				// ADD THE TOOL CALL TO THE CONVERSATION SO THE MODEL HAS
				// CONTEXT OF THE TOOL CALL.
				conversation = append(conversation, client.D{
					"role": "assistant",
					"content": fmt.Sprintf("Tool call %s: %s(%v)",
						toolCall.ID,
						toolCall.Function.Name,
						toolCall.Function.Arguments),
				})

				// WE NEED TO EXECUTE THE TOOL CALL.
				results := a.callTools(ctx, resp.Choices[0].Delta.ToolCalls)

				// NOW WE NEED TO CHECK IF THE TOOL CALLS PROVIDED ANY RESULTS
				// TO ADD TO THE CONVERSATION AND MARK WE ARE IN A TOOL CALL.
				if len(results) > 0 {
					conversation = append(conversation, results...)
					inToolCall = true
				}

			case resp.Choices[0].Delta.Content != "":
				if reasonThinking {
					reasonThinking = false
					fmt.Print("\n\n")
				}

				switch resp.Choices[0].Delta.Content {
				case "<think>":
					contentThinking = true
					continue
				case "</think>":
					contentThinking = false
					continue
				}

				switch {
				case !contentThinking:
					fmt.Print(resp.Choices[0].Delta.Content)
					chunks = append(chunks, resp.Choices[0].Delta.Content)

				case contentThinking:
					reasonContent = append(reasonContent, resp.Choices[0].Delta.Content)
					fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Content)
				}

			case resp.Choices[0].Delta.Reasoning != "":
				reasonThinking = true

				if len(reasonContent) == 0 {
					fmt.Print("\n")
				}

				reasonContent = append(reasonContent, resp.Choices[0].Delta.Reasoning)
				fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
			}
		}

		cancelContext()

		// WE NEED TO CHECK IF WE ARE IN A TOOL CALL BECAUSE WE NEED TO GIVE THE
		// MODEL THE RESULTS WITHOUT ANY NOISE. THE CHUNKS SHOULD BE EMPTY IN
		// THIS CASE BUT I DON'T TRUST MODELS.

		if !inToolCall && len(chunks) > 0 {
			fmt.Print("\n")

			content := strings.Join(chunks, " ")
			content = strings.TrimLeft(content, "\n")

			if content != "" {
				conversation = a.addToConversation(reasonContent, conversation, client.D{
					"role":    "assistant",
					"content": content,
				})
			}
		}
	}

	return nil
}

// addToConversation will add new messages to the conversation history and
// calculate the different tokens used in the conversation and display it to the
// user. It will also check the amount of input tokens currently in history
// and remove the oldest messages if we are over.
func (a *Agent) addToConversation(reasoning []string, conversation []client.D, newMessages ...client.D) []client.D {
	conversation = append(conversation, newMessages...)

	fmt.Print("\n")

	for {
		var currentWindow int
		for _, msg := range conversation {
			currentWindow += a.tke.TokenCount(msg["content"].(string))
		}

		r := strings.Join(reasoning, " ")
		reasonTokens := a.tke.TokenCount(r)

		totalTokens := currentWindow + reasonTokens
		percentage := (float64(currentWindow) / float64(contextWindow)) * 100
		of := float32(contextWindow) / float32(1024)

		fmt.Printf("\u001b[90mTokens Total[%d] Reason[%d] Window[%d] (%.0f%% of %.0fK)\u001b[0m\n", totalTokens, reasonTokens, currentWindow, percentage, of)

		// ---------------------------------------------------------------------
		// Check if we have too many input tokens and start removing messages.

		if currentWindow > contextWindow {
			fmt.Print("\u001b[90mRemoving conversation history\u001b[0m\n")
			conversation = slices.Delete(conversation, 1, 2)
			continue
		}

		break
	}

	return conversation
}

// WE NEED A FUNCTION THAT LOOKS UP THE REQUESTED TOOL BY NAME AND CALLS IT
// WITH THE MODEL PROVIDED PARAMETERS.

// callTools will lookup a requested tool by name and call it.
func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) []client.D {
	var resps []client.D

	for _, toolCall := range toolCalls {
		tool, exists := a.tools[toolCall.Function.Name]
		if !exists {
			continue
		}

		fmt.Printf("\n\n\u001b[92m%s(%v)\u001b[0m:\n\n", toolCall.Function.Name, toolCall.Function.Arguments)

		resp := tool.Call(ctx, toolCall)
		resps = append(resps, resp)

		fmt.Printf("%#v\n", resps)
	}

	return resps
}
