use anyhow::Result;
#[allow(deprecated)]
use qdrant_client::prelude::*;
use std::collections::HashMap;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, Value, STORED, TEXT};
use tantivy::{doc, Index, ReloadPolicy, TantivyDocument};

const QDRANT_URI: &str = "http://localhost:6334";

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Starting Hybrid Search Prototype (Qdrant + Tantivy)...");

    // 1. Initialize Tantivy (Keyword Search)
    println!("📚 Initializing Tantivy Index...");
    let mut schema_builder = Schema::builder();
    let title = schema_builder.add_text_field("title", TEXT | STORED);
    let body = schema_builder.add_text_field("body", TEXT);
    let id_field = schema_builder.add_u64_field("id", STORED);
    let schema = schema_builder.build();

    let index = Index::create_in_ram(schema.clone());
    let mut index_writer = index.writer(50_000_000)?;

    // Add some dummy data
    index_writer.add_document(doc!(
        title => "Rust RAG Systems",
        body => "Rust is great for RAG because of performance and safety.",
        id_field => 1u64
    ))?;
    index_writer.add_document(doc!(
        title => "Python Agents",
        body => "Python is good for prototyping agents but slow for production.",
        id_field => 2u64
    ))?;
    index_writer.commit()?;

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()?;
    let searcher = reader.searcher();

    // 2. Initialize Qdrant (Vector Search)
    println!("🔮 Connecting to Qdrant at {}...", QDRANT_URI);
    // Using deprecated client for speed in prototype, suppressing warnings would be better but let's just use it.
    #[allow(deprecated)]
    let client = QdrantClient::from_url(QDRANT_URI).build();

    // 3. Perform Hybrid Search
    let query_text = "Rust performance";
    println!("🔎 Searching for: '{}'", query_text);

    // A. Keyword Search (Tantivy)
    let query_parser = QueryParser::for_index(&index, vec![title, body]);
    let query = query_parser.parse_query(query_text)?;
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;

    let mut keyword_results: HashMap<u64, f32> = HashMap::new();
    println!("\n📄 Tantivy Results:");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let doc_id = retrieved_doc.get_first(id_field).unwrap().as_u64().unwrap();
        let doc_title = retrieved_doc.get_first(title).unwrap().as_str().unwrap();
        println!("   - [ID: {}] {} (Score: {})", doc_id, doc_title, score);
        keyword_results.insert(doc_id, score);
    }

    // B. Vector Search (Qdrant)
    let vector_results = match client {
        Ok(_c) => {
            // Mock return for prototype
            vec![
                (1, 0.95), // Rust doc (high match)
                (2, 0.40), // Python doc (low match)
            ]
        }
        Err(_) => {
            println!("   (Qdrant not reachable, using mock data)");
            vec![(1, 0.95), (2, 0.40)]
        }
    };

    println!("\n🧠 Qdrant Results (Simulated):");
    for (id, score) in &vector_results {
        println!("   - [ID: {}] Score: {}", id, score);
    }

    // 4. Reciprocal Rank Fusion (RRF)
    println!("\n⚗️  Applying Reciprocal Rank Fusion (k=60)...");
    let mut rrf_scores: HashMap<u64, f32> = HashMap::new();
    let k = 60.0;

    // Process Keyword Ranks
    let mut sorted_keyword: Vec<_> = keyword_results.iter().collect();
    sorted_keyword.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    for (rank, (id, _)) in sorted_keyword.iter().enumerate() {
        let score = 1.0 / (k + (rank as f32) + 1.0);
        *rrf_scores.entry(**id).or_insert(0.0) += score;
    }

    // Process Vector Ranks
    for (rank, (id, _)) in vector_results.iter().enumerate() {
        let score = 1.0 / (k + (rank as f32) + 1.0);
        *rrf_scores.entry(*id).or_insert(0.0) += score;
    }

    // Display Final Results
    let mut final_results: Vec<_> = rrf_scores.iter().collect();
    final_results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\n🏆 Final Hybrid Results:");
    for (id, score) in final_results {
        println!("   - Document ID: {} | RRF Score: {:.4}", id, score);
    }

    Ok(())
}
