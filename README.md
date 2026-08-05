# AI Document Retrieval & QA Assistant

A full-stack RAG (Retrieval-Augmented Generation) application that lets users upload PDFs, index them into a vector database, and ask questions against the document content. Built with LangGraph, LangChain, Next.js 14, and Supabase pgvector.

Based on concepts from [Learning LangChain (O'Reilly)](https://www.oreilly.com/library/view/learning-langchain/9781098167271/).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Switching LLM Providers](#switching-llm-providers)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [CI](#ci)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This is a monorepo containing two workspaces:

- **`backend/`** — A LangGraph server exposing two stateful graphs: one for document ingestion and one for retrieval + response generation.
- **`frontend/`** — A Next.js 14 app providing a chat UI with PDF upload, real-time streaming responses, and source citations.

The core workflow:

1. User uploads a PDF through the frontend.
2. The PDF is parsed into page-level documents, embedded using Google's `text-embedding-004`, and stored in Supabase pgvector.
3. User asks a question. An LLM-based router decides whether to retrieve documents or answer directly.
4. If retrieval is needed, the system pulls the top-k relevant chunks from the vector store, passes them as context to the LLM, and streams the response back token-by-token via SSE.

---

## Architecture

```
                         Frontend (Next.js 14)
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │   Chat UI ──> /api/chat (SSE)     /api/ingest (POST)    │
  │                  │                      │               │
  └──────────────────┼──────────────────────┼───────────────┘
                     │                      │
          ┌──────────▼──────────────────────▼───────────┐
          │         LangGraph Server (Backend)          │
          │                                             │
          │   Retrieval Graph          Ingestion Graph  │
          │   ┌────────────────┐       ┌────────────┐   │
          │   │ checkQueryType │       │ ingestDocs │   │
          │   └───┬────────┬───┘       └────────────┘   │
          │       │        │                            │
          │   retrieve   direct                         │
          │       │      answer                         │
          │       ▼                                     │
          │   generateResponse                          │
          └────────────────┬────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Supabase   │
                    │  (pgvector) │
                    └─────────────┘
```

### Retrieval Graph

The retrieval graph handles user queries through the following nodes:

1. **`checkQueryType`** — Uses structured output (Zod schema) to classify the query as `retrieve` or `direct`.
2. **`retrieveDocuments`** — Performs a similarity search against Supabase pgvector, returns deduplicated top-k results.
3. **`generateResponse`** — Formats retrieved documents as XML context, injects them into the prompt, and generates a response.
4. **`directAnswer`** — Answers general knowledge questions without document context.

Routing is handled via a conditional edge from `checkQueryType`.

### Ingestion Graph

A single-node graph that takes parsed `Document` objects, embeds them, and upserts them into the Supabase `documents` table.

---

## Tech Stack

**Backend:**
[LangGraph](https://langchain-ai.github.io/langgraphjs/) |
[LangChain](https://js.langchain.com) |
[Google Gemini](https://ai.google.dev) (default LLM) |
[Supabase pgvector](https://supabase.com/docs/guides/ai/vector-columns) |
[Zod](https://zod.dev) |
TypeScript

**Frontend:**
[Next.js 14](https://nextjs.org) |
[React 18](https://react.dev) |
[TailwindCSS](https://tailwindcss.com) |
[Radix UI](https://www.radix-ui.com) / [shadcn/ui](https://ui.shadcn.com) |
[Lucide](https://lucide.dev) |
[Geist](https://vercel.com/font)

**Tooling:**
[Turborepo](https://turbo.build) |
[Yarn Workspaces](https://classic.yarnpkg.com/en/docs/workspaces/) |
[Jest](https://jestjs.io) |
[ESLint](https://eslint.org) + [Prettier](https://prettier.io) |
[GitHub Actions](.github/workflows/ci.yml) |
[LangSmith](https://smith.langchain.com) (optional tracing)

---

## Project Structure

```
.
├── backend/
│   ├── src/
│   │   ├── ingestion_graph/
│   │   │   ├── graph.ts            # Ingestion graph definition
│   │   │   ├── state.ts            # State schema for document indexing
│   │   │   └── configuration.ts    # Config: docs file path, sample docs toggle
│   │   ├── retrieval_graph/
│   │   │   ├── graph.ts            # Retrieval graph with routing + generation
│   │   │   ├── state.ts            # Agent state: query, route, messages, documents
│   │   │   ├── configuration.ts    # Config: model selection
│   │   │   ├── prompts.ts          # System prompts for router and responder
│   │   │   └── utils.ts            # Document formatting (XML)
│   │   └── shared/
│   │       ├── configuration.ts    # Base config: retriever provider, k, filters
│   │       ├── retrieval.ts        # Supabase vector store retriever
│   │       ├── state.ts            # Document reducer (dedup, merge, delete)
│   │       └── utils.ts            # Multi-provider LLM loader
│   ├── __tests__/                  # Unit + integration tests
│   ├── langgraph.json              # Graph registry for LangGraph CLI
│   ├── demo.ts                     # Standalone demo script
│   └── .env.example
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx                # Main chat page
│   │   ├── layout.tsx              # Root layout
│   │   └── api/
│   │       ├── chat/route.ts       # SSE streaming endpoint
│   │       └── ingest/route.ts     # PDF upload + ingestion endpoint
│   ├── components/
│   │   ├── chat-message.tsx        # Chat bubble with source accordion
│   │   ├── example-prompts.tsx     # Starter prompt cards
│   │   ├── file-preview.tsx        # Uploaded file preview
│   │   └── ui/                     # shadcn/ui components
│   ├── lib/
│   │   ├── langgraph-base.ts       # Base client class (thread management)
│   │   ├── langgraph-client.ts     # Frontend client singleton
│   │   ├── langgraph-server.ts     # Server-side client singleton
│   │   ├── pdf.ts                  # PDF parsing via PDFLoader
│   │   └── utils.ts
│   ├── constants/graphConfigs.ts   # Default graph configs
│   ├── types/graphTypes.ts         # TypeScript type definitions
│   └── .env.example
│
├── scripts/
│   └── checkLanggraphPaths.js      # Validates langgraph.json against source files
│
├── .github/workflows/ci.yml       # Lint + format CI checks
├── turbo.json                      # Turborepo config
└── package.json                    # Root monorepo config (Yarn Workspaces)
```

---

## Getting Started

### Prerequisites

- Node.js 20+
- Yarn 1.22+
- A [Supabase](https://supabase.com) project (free tier works)
- A [Google AI API key](https://aistudio.google.com/apikey)

### 1. Clone and install

```bash
git clone https://github.com/Murthypsty0419/AI-Document-Retrieval-QA-Assistant.git
cd AI-Document-Retrieval-QA-Assistant
yarn install
```

### 2. Set up Supabase

Run these in your Supabase SQL Editor:

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the documents table
CREATE TABLE documents (
  id bigserial PRIMARY KEY,
  content text,
  metadata jsonb,
  embedding vector(768)
);

-- Create the similarity search function
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding vector(768),
  match_count int DEFAULT 5,
  filter jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE documents.metadata @> filter
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

The embedding dimension is 768 because the project uses Google's `text-embedding-004`.

### 3. Configure environment variables

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

**`backend/.env`**

```env
GOOGLE_API_KEY=<your-google-ai-api-key>
SUPABASE_URL=<your-supabase-project-url>
SUPABASE_SERVICE_ROLE_KEY=<your-supabase-service-role-key>
SUPABASE_MATCH_FUNCTION=match_documents

# Optional
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-api-key>
LANGCHAIN_PROJECT=ai-agent-pdf-chatbot
```

**`frontend/.env`**

```env
NEXT_PUBLIC_LANGGRAPH_API_URL=http://localhost:2024
LANGCHAIN_API_KEY=<your-langsmith-api-key>
LANGGRAPH_INGESTION_ASSISTANT_ID=ingestion_graph
LANGGRAPH_RETRIEVAL_ASSISTANT_ID=retrieval_graph

# Optional
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=pdf-chatbot
```

### 4. Start the backend

```bash
cd backend
npx @langchain/langgraph-cli dev
```

This starts the LangGraph dev server on `http://localhost:2024`, serving both graphs defined in [`langgraph.json`](backend/langgraph.json).

### 5. Start the frontend

In a new terminal:

```bash
cd frontend
yarn dev
```

Open `http://localhost:3000`. Upload a PDF, then ask questions about it.

---

## Switching LLM Providers

The default model is `google-genai/gemini-2.5-flash-lite`. To use a different provider, update `queryModel` in [`frontend/constants/graphConfigs.ts`](frontend/constants/graphConfigs.ts):

```typescript
export const retrievalAssistantStreamConfig = {
  queryModel: 'openai/gpt-4o', // change this
  retrieverProvider: 'supabase',
  k: 5,
};
```

Supported providers and example values:

| Provider | `queryModel` value |
|---|---|
| Google Gemini | `google-genai/gemini-2.5-flash-lite` |
| OpenAI | `openai/gpt-4o` |
| Anthropic | `anthropic/claude-sonnet-4-20250514` |
| Ollama | `ollama/llama3` |
| Groq | `groq/llama-3.1-70b-versatile` |
| Together | `together/meta-llama/Llama-3-70b-chat-hf` |
| Fireworks | `fireworks/accounts/fireworks/models/llama-v3-70b-instruct` |
| Mistral | `mistralai/mistral-large-latest` |
| Bedrock | `bedrock/anthropic.claude-3-sonnet-20240229-v1:0` |
| Cohere | `cohere/command-r-plus` |
| Vertex AI | `google-vertexai/gemini-pro` |
| DeepSeek | `deepseek/deepseek-chat` |
| Cerebras | `cerebras/llama3.1-70b` |
| xAI | `xai/grok-beta` |
| Azure OpenAI | `azure_openai/<deployment-name>` |

Each provider needs its own API key as an env variable (e.g., `OPENAI_API_KEY`). See the [LangChain JS integrations docs](https://js.langchain.com/docs/integrations/chat/) for details.

---

## API Reference

### POST `/api/chat`

Streams an AI response for a given query.

**Request:**

```json
{
  "message": "What is this document about?",
  "threadId": "<thread-id>"
}
```

**Response:** SSE stream (`text/event-stream`)

Each event is `data: <json>\n\n` with the following event types:

| Event | Payload |
|---|---|
| `messages/partial` | Partial AI response tokens for streaming display |
| `updates` | Graph node outputs, including `retrieveDocuments.documents` for source citations |

### POST `/api/ingest`

Uploads and indexes PDF files.

**Request:** `multipart/form-data` with a `files` field.

**Constraints:**

- Max 5 files per request
- Max 10 MB per file
- PDF only (`application/pdf`)

**Response:**

```json
{
  "message": "Documents ingested successfully",
  "threadId": "<thread-id>"
}
```

---

## Configuration

Configuration is passed to the LangGraph graphs via `configurable` at runtime.

### Retrieval graph

| Key | Type | Default | Notes |
|---|---|---|---|
| `queryModel` | `string` | `google-genai/gemini-2.5-flash-lite` | Format: `provider/model-name` |
| `retrieverProvider` | `string` | `supabase` | Vector store backend |
| `k` | `number` | `5` | Number of documents to retrieve |
| `filterKwargs` | `object` | `{}` | Metadata filters for search |

### Ingestion graph

| Key | Type | Default | Notes |
|---|---|---|---|
| `retrieverProvider` | `string` | `supabase` | Vector store backend |
| `useSampleDocs` | `boolean` | `false` | Index bundled sample docs instead of uploaded PDFs |
| `docsFile` | `string` | `./src/sample_docs.json` | Path to sample docs file |

---

## Environment Variables

### Backend

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google AI key (Gemini LLM + embeddings) |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service role key |
| `SUPABASE_MATCH_FUNCTION` | No | Match function name (default: `match_documents`) |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |

### Frontend

| Variable | Required | Description |
|---|---|---|
| `NEXT_PUBLIC_LANGGRAPH_API_URL` | Yes | LangGraph server URL |
| `LANGCHAIN_API_KEY` | Yes | API key for server-side LangGraph calls |
| `LANGGRAPH_INGESTION_ASSISTANT_ID` | Yes | Ingestion graph ID |
| `LANGGRAPH_RETRIEVAL_ASSISTANT_ID` | Yes | Retrieval graph ID |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |

---

## Testing

```bash
# Backend unit tests
cd backend && yarn test

# Backend integration tests (needs env vars configured)
cd backend && yarn test:int

# Watch mode
cd backend && yarn test:watch

# Coverage
cd backend && yarn test:coverage

# Frontend tests
cd frontend && yarn test
```

### Linting and formatting

```bash
# From root
yarn lint
yarn format
yarn format:check

# Validate langgraph.json paths point to real exports
cd backend && yarn lint:langgraph-json
```

---

## CI

GitHub Actions runs on every push to `main` and on all PRs (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

- **Format check** — Prettier
- **Lint check** — ESLint + `langgraph.json` path validation

---

## Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run `yarn format && yarn lint` to ensure everything passes
5. Open a PR
