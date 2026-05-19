-- 1. Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the talent_embeddings table
-- 'd' here is the dimensionality of the embeddings. Gemini models typically use 768 or 1024.
-- Since the frontend uses flash for analysis, let's assume 768 dimensions for standard text embeddings.
CREATE TABLE IF NOT EXISTS talent_embeddings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    profile_id TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    sustainability_index FLOAT DEFAULT 0.0,
    agent_stability FLOAT DEFAULT 0.0,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 3. Create the HNSW index for O(d * log N) retrieval complexity
-- 'm' determines the max connections per node.
-- 'ef_construction' determines the size of the dynamic candidate list during construction.
CREATE INDEX ON talent_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);

-- 4. Create the "Sober and Durable" matching RPC function
-- This function uses the index to quickly find the closest candidates,
-- then reranks them by applying the Sustainability Index and Agent Stability.
CREATE OR REPLACE FUNCTION match_talent(
    query_embedding VECTOR(768),
    match_threshold FLOAT,
    match_count INT,
    sustainability_weight FLOAT DEFAULT 0.1,
    stability_weight FLOAT DEFAULT 0.1
)
RETURNS TABLE (
    id UUID,
    profile_id TEXT,
    sustainability_index FLOAT,
    agent_stability FLOAT,
    metadata JSONB,
    base_similarity FLOAT,
    final_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH candidates AS (
        SELECT 
            t.id,
            t.profile_id,
            t.sustainability_index,
            t.agent_stability,
            t.metadata,
            1 - (t.embedding <=> query_embedding) AS base_similarity
        FROM talent_embeddings t
        ORDER BY t.embedding <=> query_embedding
        LIMIT match_count * 2 -- fetch twice the match_count to allow meaningful re-ranking
    )
    SELECT 
        c.id,
        c.profile_id,
        c.sustainability_index,
        c.agent_stability,
        c.metadata,
        c.base_similarity,
        -- The "Sober & Durable" Equation:
        (c.base_similarity + (c.sustainability_index * sustainability_weight) + (c.agent_stability * stability_weight)) AS final_score
    FROM candidates c
    WHERE c.base_similarity >= match_threshold
    ORDER BY final_score DESC
    LIMIT match_count;
END;
$$;
