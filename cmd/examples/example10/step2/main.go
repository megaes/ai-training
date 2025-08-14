// https://ampcode.com/how-to-build-an-agent
//
// This example shows you how add token counting, context window limits, and
// better UI formatting to the chat agent from step 1.
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

// Agent represents the chat agent that can use tools to perform tasks.
type Agent struct {
	sseClient      *client.SSEClient[client.ChatSSE]
	getUserMessage func() (string, bool)

	// WE WILL ADD OUR OWN TOKENIZER TO COUNT THE TOKENS IN THE CONVERSATION.

	tke *tiktoken.Tiktoken
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

	// WE WILL CONSTRUCT OUR OWN TOKENIZER.
	tke, err := tiktoken.NewTiktoken()
	if err != nil {
		return nil, fmt.Errorf("failed to create tiktoken: %w", err)
	}

	agent := Agent{
		sseClient:      sseClient,
		getUserMessage: getUserMessage,

		// ADD THE TOKENIZER TO THE AGENT.
		tke: tke,
	}

	return &agent, nil
}

// WE WILL ADD A SYSTEM PROMPT TO THE AGENT TO HELP WITH CLARIFYING INSTRUCTIONS.

// The system prompt for the model so it behaves as expected.
var systemPrompt = `You are a helpful coding assistant that has tools to assist
you in coding.

Reasoning: high
`

// Run starts the agent and runs the chat loop.
func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D

	// WE WILL MAINTAIN THE REASONING CONTENT FOR TOKEN COUNTING.
	// AND TO MAKE SURE WE DON'T ADD THE REASONING TO THE CONVERSATION.
	var reasonContent []string

	// WE WILL ADD THE SYSTEM PROMPT TO THE CONVERSATION.
	conversation = append(conversation, client.D{
		"role":    "system",
		"content": systemPrompt,
	})

	fmt.Printf("\nChat with %s (use 'ctrl-c' to quit)\n", model)

	for {
		fmt.Print("\u001b[94m\nYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		conversation = append(conversation, client.D{
			"role":    "user",
			"content": userInput,
		})

		d := client.D{
			"model":       model,
			"messages":    conversation,
			"max_tokens":  contextWindow,
			"temperature": 0.0,
			"top_p":       0.1,
			"top_k":       1,
			"stream":      true,
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

		// WE WILL CREATE FLAGS TO KNOW WHEN WE ARE PROCESSING REASONING CONTENT.

		reasonThinking := false  // GPT models provide a Reasoning field.
		contentThinking := false // Other reasoning models use <think> tags.
		reasonContent = nil      // Reset the reasoning content for this next call.

		// WE WILL ADD SOME IMPROVED FORMATTING.
		fmt.Print("\n")

		for resp := range ch {
			switch {
			case resp.Choices[0].Delta.Content != "":

				// WE NEED TO RESET THE REASONING FLAG ONCE THE MODEL IS
				// DONE REASONING.
				if reasonThinking {
					reasonThinking = false
					fmt.Print("\n\n")
				}

				// WE NEED TO CHECK IF THE REASONING IS HAPPENING VIA
				// <think> TAGS.
				switch resp.Choices[0].Delta.Content {
				case "<think>":
					contentThinking = true
					continue
				case "</think>":
					contentThinking = false
					continue
				}

				// WE NEED TO ADJUST OUR ORIGINAL SWITCH TO TAKE INTO ACCOUNT
				// WE MIGHT HAVE BEEN PROCESSING <think> TAGS.
				switch {
				case !contentThinking:
					fmt.Print(resp.Choices[0].Delta.Content)
					chunks = append(chunks, resp.Choices[0].Delta.Content)

				case contentThinking:
					reasonContent = append(reasonContent, resp.Choices[0].Delta.Content)
					fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Content)
				}

			// WE NEED TO CHECK IF THE MODEL IS THINKING VIA THIS REASONING
			// FIELD AND TRACK AND CAPTURE THAT SEPARATELY FROM THE CONVERSATION.
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

		if len(chunks) > 0 {
			fmt.Print("\n")

			// REMOVING <think> TAGS FROM THE CONTENT WILL LEAVE EXTRA CRLF
			// CHARACTERS WE NEED TO REMOVE.
			content := strings.Join(chunks, " ")
			content = strings.TrimLeft(content, "\n")

			// WE NEED TO CHECK IF THE CONTENT IS EMPTY AFTER REMOVING CRLF.
			if content != "" {

				// WE WILL USE THIS NEW FUNCTION THAT WILL HANDLE TOKEN COUNTING.
				conversation = a.addToConversation(reasonContent, conversation, client.D{
					"role":    "assistant",
					"content": content,
				})
			}
		}
	}

	return nil
}

// WE WILL ADD THIS NEW FUNCTION THAT WILL ADD MESSAGE TO THE CONVERSATION
// HISTORY AND CALCULATE THE TOKENS USED IN THE CONVERSATION. IF WE REACH
// THE CONTEXT WINDOW WE WILL REMOVE THE OLDEST MESSAGES.

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
