// This example shows you how to use the Llama3.2 vision model to generate
// an image description and update the image with the description.
// After storing the image embeddings into the database, let's search for it.
//
// # Running the example:
//
//	$ make example9-step4
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"github.com/dsoprea/go-exif/v3"
	exifcommon "github.com/dsoprea/go-exif/v3/common"
	jpg "github.com/dsoprea/go-jpeg-image-structure/v2"
	pis "github.com/dsoprea/go-png-image-structure/v2"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type document struct {
	ID        string    `bson:"id"`
	Embedding []float64 `bson:"embedding"`
}

type searchResult struct {
	ID        int       `bson:"id"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
	Score     float64   `bson:"score"`
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	fileName := "cmd/samples/roseimg.png"

	data, err := readImage(fileName)
	if err != nil {
		return fmt.Errorf("read image: %w", err)
	}

	prompt := "Describe the image and be concise and accurate."

	var mimeType string
	switch filepath.Ext(fileName) {
	case ".jpg", ".jpeg":
		mimeType = "image/jpg"
	case ".png":
		mimeType = "image/png"
	default:
		return fmt.Errorf("unsupported file type: %s", filepath.Ext(fileName))
	}

	// -------------------------------------------------------------------------

	fmt.Println("Generating image description...")

	llm, err := ollama.New(
		ollama.WithModel("llama3.2-vision"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.BinaryContent{
					MIMEType: mimeType,
					Data:     data,
				},
				llms.TextContent{
					Text: prompt,
				},
			},
		},
	}

	cr, err := llm.GenerateContent(context.Background(), messages)
	if err != nil {
		return fmt.Errorf("generate content: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Printf("Updating Image description: %s\n", cr.Choices[0].Content)

	err = updateImage(fileName, cr.Choices[0].Content)
	if err != nil {
		return fmt.Errorf("update image: %w", err)
	}

	fmt.Printf("Inserting image description into the database: %s\n", cr.Choices[0].Content)

	vector, err := generateEmbeddings(cr.Choices[0].Content)
	if err != nil {
		return fmt.Errorf("generate embeddings: %w", err)
	}

	err = updateDatabase(fileName, vector)
	if err != nil {
		return fmt.Errorf("update database: %w", err)
	}

	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Search the database for an image: ")

	question, _ := reader.ReadString('\n')
	if question == "" {
		return nil
	}

	fmt.Print("THIS MAY TAKE A MINUTE OR MORE, BE PATIENT\n\n")

	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()

	results, err := vectorSearch(ctx, question)
	if err != nil {
		return fmt.Errorf("vectorSearch: %w", err)
	}

	if err := questionResponse(ctx, question, results); err != nil {
		return fmt.Errorf("questionResponse: %w", err)
	}

	return nil
}

func readImage(fileName string) ([]byte, error) {
	f, err := os.OpenFile(fileName, os.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	return data, nil
}

func updateImage(fileName string, description string) error {
	im, err := exifcommon.NewIfdMappingWithStandard()
	if err != nil {
		return fmt.Errorf("new idf mapping: %w", err)
	}

	ti := exif.NewTagIndex()
	ib := exif.NewIfdBuilder(im, ti, exifcommon.IfdStandardIfdIdentity, exifcommon.EncodeDefaultByteOrder)

	err = ib.AddStandardWithName("ImageDescription", description)
	if err != nil {
		return fmt.Errorf("add standard: %w", err)
	}

	// -------------------------------------------------------------------------

	switch filepath.Ext(fileName) {
	case ".jpg", ".jpeg":
		intfc, err := jpg.NewJpegMediaParser().ParseFile(fileName)
		if err != nil {
			return fmt.Errorf("parse file: %w", err)
		}

		cs := intfc.(*jpg.SegmentList)
		err = cs.SetExif(ib)
		if err != nil {
			return fmt.Errorf("set ib: %w", err)
		}

		f, err := os.Create(fileName)
		if err != nil {
			return fmt.Errorf("create: %w", err)
		}

		err = cs.Write(f)
		if err != nil {
			return fmt.Errorf("wrtite: %w", err)
		}
		defer f.Close()

	case ".png":
		intfc, err := pis.NewPngMediaParser().ParseFile(fileName)
		if err != nil {
			return fmt.Errorf("parse file: %w", err)
		}

		cs := intfc.(*pis.ChunkSlice)
		err = cs.SetExif(ib)
		if err != nil {
			return fmt.Errorf("set ib: %w", err)
		}

		f, err := os.Create(fileName)
		if err != nil {
			return fmt.Errorf("create: %w", err)
		}

		err = cs.WriteTo(f)
		if err != nil {
			return fmt.Errorf("wrtite: %w", err)
		}
		defer f.Close()

	default:
		return fmt.Errorf("unsupported file type: %s", filepath.Ext(fileName))
	}

	return nil
}

func generateEmbeddings(description string) ([][]float32, error) {
	llm, err := ollama.New(
		ollama.WithModel("mxbai-embed-large"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		log.Fatal(err)
	}

	vectors, err := llm.CreateEmbedding(context.Background(), []string{description})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Received embeddings from model: %v\n", vectors)

	return vectors, nil
}

func updateDatabase(fileName string, vectors [][]float32) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// -------------------------------------------------------------------------
	// Connect to mongo

	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return fmt.Errorf("connectToMongo: %w", err)
	}
	defer client.Disconnect(ctx)

	fmt.Println("Connected to MongoDB")

	// -------------------------------------------------------------------------
	// Create database and collection

	const dbName = "example9"
	const collectionName = "images"

	db := client.Database(dbName)

	col, err := mongodb.CreateCollection(ctx, db, collectionName)
	if err != nil {
		return fmt.Errorf("createCollection: %w", err)
	}

	fmt.Println("Created Collection")

	// -------------------------------------------------------------------------
	// Create vector index

	const indexName = "vector_index"

	settings := mongodb.VectorIndexSettings{
		NumDimensions: len(vectors[0]),
		Path:          "embedding",
		Similarity:    "cosine",
	}

	if err := mongodb.CreateVectorIndex(ctx, col, indexName, settings); err != nil {
		return fmt.Errorf("createVectorIndex: %w", err)
	}

	fmt.Println("Created Vector Index")

	// -------------------------------------------------------------------------
	// Store some documents with their embeddings.

	if err := storeDocuments(ctx, col, fileName, vectors); err != nil {
		return fmt.Errorf("storeDocuments: %w", err)
	}

	return nil
}

func storeDocuments(ctx context.Context, col *mongo.Collection, fileName string, vectors [][]float32) error {

	// If these records already exist, we don't need to add them again.
	findRes, err := col.Find(ctx, bson.D{})
	if err != nil {
		return fmt.Errorf("find: %w", err)
	}
	defer findRes.Close(ctx)

	if findRes.RemainingBatchLength() != 0 {
		return nil
	}

	// -------------------------------------------------------------------------

	// Apply a unique index just to be safe.
	unique := true
	indexModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "id", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}
	col.Indexes().CreateOne(ctx, indexModel)

	// -------------------------------------------------------------------------

	// Let's add two documents to the database.

	d1 := document{
		ID:        fileName,
		Embedding: float32ToFloat64(vectors[0]),
	}

	res, err := col.InsertMany(ctx, []any{d1})
	if err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	fmt.Println(res.InsertedIDs)

	return nil
}

func vectorSearch(ctx context.Context, question string) ([]searchResult, error) {

	// -------------------------------------------------------------------------
	// Use ollama to generate a vector embedding for the question.

	// Open a connection with ollama to access the model.
	llm, err := ollama.New(
		ollama.WithModel("mxbai-embed-large"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}

	// Get the vector embedding for the question.
	embedding, err := llm.CreateEmbedding(context.Background(), []string{question})
	if err != nil {
		return nil, fmt.Errorf("create embedding: %w", err)
	}

	// -------------------------------------------------------------------------
	// Establish a connection with mongo and access the collection.

	// Connect to mongodb.
	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return nil, fmt.Errorf("connectToMongo: %w", err)
	}

	const dbName = "example9"
	const collectionName = "images"

	// Capture a connection to the collection. We assume this exists with
	// data already.
	col := client.Database(dbName).Collection(collectionName)

	// -------------------------------------------------------------------------
	// Perform the vector search.

	// We want to find the nearest neighbors from the question vector embedding.
	pipeline := mongo.Pipeline{
		{{
			Key: "$vectorSearch",
			Value: bson.M{
				"index":         "vector_index",
				"exact":         false,
				"path":          "embedding",
				"queryVector":   float32ToFloat64(embedding[0]),
				"numCandidates": 5,
				"limit":         5,
			}},
		},
		{{
			Key: "$project",
			Value: bson.M{
				"id":        "",
				"embedding": []float64{},
				"score": bson.M{
					"$meta": "vectorSearchScore",
				},
			}},
		},
	}

	cur, err := col.Aggregate(ctx, pipeline)
	if err != nil {
		return nil, fmt.Errorf("aggregate: %w", err)
	}
	defer cur.Close(ctx)

	var results []searchResult
	if err := cur.All(ctx, &results); err != nil {
		return nil, fmt.Errorf("all: %w", err)
	}

	return results, nil
}

func questionResponse(ctx context.Context, question string, results []searchResult) error {

	// Open a connection with ollama to access the model.
	llm, err := ollama.New(
		ollama.WithModel("llama3.2"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	// Format a prompt to direct the model what to do with the content and
	// the question.
	prompt := `Use the following pieces of information to answer the user's question.
	
If you don't know the answer, say that you don't know.	

Answer the question and provide additional helpful information.

Responses should be properly formatted to be easily read.
	
Context: %s
	
Question: %s
`

	// var wordCount int
	var chunks strings.Builder

	for _, res := range results {
		if res.Score >= .70 {
			chunks.WriteString(res.Text)
			chunks.WriteString(".\n")
		}
	}

	content := chunks.String()
	if content == "" {
		fmt.Println("Don't have enough information to provide an answer")
		return nil
	}

	finalPrompt := fmt.Sprintf(prompt, content, question)

	// Setup a wait group to wait for the entire response.
	var wg sync.WaitGroup
	wg.Add(1)

	// This function will display the response as it comes from the server.
	f := func(ctx context.Context, chunk []byte) error {
		if ctx.Err() != nil || len(chunk) == 0 {
			wg.Done()
			return nil
		}

		fmt.Printf("%s", chunk)
		return nil
	}

	// Send the prompt to the model server.
	if _, err := llm.Call(ctx, finalPrompt, llms.WithStreamingFunc(f)); err != nil {
		return fmt.Errorf("call: %w", err)
	}

	// Wait until we receive the entire response.
	wg.Wait()

	return nil
}

func float32ToFloat64(src []float32) []float64 {
	dst := make([]float64, len(src))
	for i, v := range src {
		dst[i] = float64(v)
	}
	return dst
}
