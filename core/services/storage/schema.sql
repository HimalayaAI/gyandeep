-- Gyandeep minimal database schema
-- Focused on personalized learning + pgvector embeddings

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS students (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT,
    grade INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS books (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    file_hash TEXT,
    total_pages INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS books_file_hash_idx ON books(file_hash);

CREATE TABLE IF NOT EXISTS ocr_pages (
    id BIGSERIAL PRIMARY KEY,
    book_id UUID REFERENCES books(id) ON DELETE CASCADE,
    page_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (book_id, page_index)
);

CREATE INDEX IF NOT EXISTS ocr_pages_book_page_idx ON ocr_pages(book_id, page_index);

CREATE TABLE IF NOT EXISTS learning_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID REFERENCES students(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    prompt TEXT,
    response TEXT,
    score REAL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS text_chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source, chunk_index)
);

CREATE INDEX IF NOT EXISTS text_chunks_embedding_idx
    ON text_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
