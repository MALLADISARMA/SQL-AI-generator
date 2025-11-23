# app.py
import os
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
import tempfile
import pathlib
import re
from typing import List, Tuple, Dict, Any

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY missing. Please add it inside .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"
TABLE_NAME = "user_data"

# ---------------------------------------------------
# Helper: Save Uploaded File
# ---------------------------------------------------
def save_uploaded_file(uploaded_file) -> str:
    suffix = pathlib.Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name

# ---------------------------------------------------
# Load CSV/Excel into SQLite with Enhanced Processing
# ---------------------------------------------------
def load_file_into_sqlite(filepath: str, db_path: str, table_name: str) -> sqlite3.Connection:
    # read file
    if filepath.lower().endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # Clean column names
    df.columns = [str(c).strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns]

    # Handle missing values in a safe way (keep numeric NaNs where possible,
    # but fill object columns to avoid SQL issues)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("")

    # ensure directory exists for DB file
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    # write to sqlite
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    # Create simple indexes for text columns (safe attempt)
    cur = conn.cursor()
    for col in df.columns:
        # Only create index for reasonable column names (alphanumeric + underscore)
        if re.match(r"^[A-Za-z_]\w*$", col):
            idx_name = f"idx_{table_name}_{col}"
            try:
                cur.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}"("{col}")')
            except Exception:
                # ignore any index creation errors
                pass
    conn.commit()

    return conn

# ---------------------------------------------------
# Get Enhanced Schema with Sample Data
# ---------------------------------------------------
def get_enhanced_schema(conn: sqlite3.Connection, table_name: str) -> Tuple[str, str, List[str], Dict[str, List[str]]]:
    cur = conn.cursor()

    # Get CREATE TABLE statement safely
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    row = cur.fetchone()
    schema = row[0] if row and row[0] else f"Table '{table_name}' not found."

    # Get column info
    cur.execute(f"PRAGMA table_info('{table_name}')")
    columns_info = cur.fetchall()
    columns = [col[1] for col in columns_info] if columns_info else []

    # Sample values for each column
    sample_data: Dict[str, List[str]] = {}
    for col in columns:
        try:
            cur.execute(f"SELECT DISTINCT \"{col}\" FROM \"{table_name}\" WHERE \"{col}\" != '' LIMIT 5")
            samples = [str(r[0]) for r in cur.fetchall()]
        except Exception:
            samples = []
        sample_data[col] = samples

    # Unique values for low-cardinality columns
    unique_values: Dict[str, List[str]] = {}
    for col in columns:
        try:
            cur.execute(f"SELECT COUNT(DISTINCT \"{col}\") FROM \"{table_name}\"")
            distinct_count_row = cur.fetchone()
            distinct_count = distinct_count_row[0] if distinct_count_row else 0
        except Exception:
            distinct_count = 0

        if distinct_count and distinct_count <= 100:
            try:
                cur.execute(f"SELECT DISTINCT \"{col}\" FROM \"{table_name}\" WHERE \"{col}\" != ''")
                unique_values[col] = [str(r[0]) for r in cur.fetchall()]
            except Exception:
                unique_values[col] = []

    # Schema description for AI context
    schema_description = f"Table: {table_name}\nColumns:\n"
    for col in columns:
        samples = ", ".join(sample_data.get(col, [])[:3])
        schema_description += f"  - {col}: Sample values: [{samples}]\n"
        if col in unique_values and unique_values[col]:
            schema_description += f"    All unique values: {unique_values[col]}\n"

    return schema, schema_description, columns, unique_values

# ---------------------------------------------------
# Extract Keywords and Names from User Question
# ---------------------------------------------------
def extract_keywords(question: str, unique_values: Dict[str, List[str]]) -> Dict[str, List[str]]:
    matches: Dict[str, List[str]] = {}
    question_lower = question.lower()

    for col, values in unique_values.items():
        for value in values:
            value_lower = str(value).lower()
            # exact token or phrase match (word boundary)
            if value_lower and (value_lower in question_lower or question_lower in value_lower):
                matches.setdefault(col, [])
                if value not in matches[col]:
                    matches[col].append(value)
    return matches

# ---------------------------------------------------
# Enhanced Gemini: Convert NL → SQL with Context
# ---------------------------------------------------
def generate_sql_query(question: str, schema_sql: str, schema_desc: str,
                       columns: List[str], conn: sqlite3.Connection, unique_values: Dict[str, List[str]]) -> str | None:

    # Identify matches from question to actual values
    keyword_matches = extract_keywords(question, unique_values)

    context = ""
    if keyword_matches:
        context = "\n\nIMPORTANT - The user mentioned these values which exist in the data:\n"
        for col, values in keyword_matches.items():
            context += f"  - Column '{col}' contains: {values}\n"
            context += f"    Use EXACT case-sensitive matching for these values.\n"

    system_instruction = f"""You are an expert SQLite query generator. Your task is to convert natural language questions into precise SQL queries.

CRITICAL RULES:
1. Table name is ALWAYS '{TABLE_NAME}'
2. Return ONLY the SQL query - no explanations, no markdown, no comments
3. Do NOT include semicolon at the end
4. Use CASE-SENSITIVE exact matching for names and text values when provided in context
5. For text searches with words like 'has', 'contains', 'includes', use LIKE with wildcards
6. Use proper aggregation (COUNT, SUM, AVG) when question implies it
7. Use GROUP BY when grouping is implied
8. Use ORDER BY when sorting is implied (highest, lowest, top, bottom)
9. For "all" or "list" questions, select relevant columns only
10. If you refer to columns, use the exact column names provided below
"""

    prompt = f"""Database Schema (short):
{schema_desc}

{context}
User Question: {question}

Generate the SQL query (SQLite compatible). Return only the SQL query.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                top_p=0.95
            )
        )
        sql = response.text.strip()

        # remove possible code fences, trailing semicolons and whitespace
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s*```$", "", sql)
        sql = sql.rstrip(";").strip()

        # safety: ensure query references the table or columns (basic check)
        lowered = sql.lower()
        if not lowered.startswith("select"):
            return None

        return sql
    except Exception as e:
        st.error(f"Gemini SQL generation error: {e}")
        return None

# ---------------------------------------------------
# Enhanced Gemini: Natural Language Summary
# ---------------------------------------------------
def generate_nl_summary(question: str, sql_query: str, sql_result: List[Tuple[Any, ...]], columns: List[str]) -> str:
    # Build a concise preview
    if sql_result and columns:
        df_preview = pd.DataFrame(sql_result, columns=columns)
        if len(df_preview) > 10:
            result_preview = f"First 10 rows of {len(df_preview)} total results:\n{df_preview.head(10).to_string(index=False)}"
        else:
            result_preview = df_preview.to_string(index=False)
    else:
        result_preview = "No results found."

    system_instruction = """You are a helpful data analyst. Create a clear, concise answer to the user's question based on the SQL results.

RULES:
1. Start with the main finding.
2. Use plain language and specific values from results.
3. If no results, say so clearly.
4. Keep it short (1-3 sentences).
"""

    prompt = f"""User's Question: {question}

SQL Query Used: {sql_query}

Results:
{result_preview}

Provide a short natural language answer based only on the results:
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.25,
                top_p=0.95
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Could not generate summary: {e}"

# ---------------------------------------------------
# Query Optimization
# ---------------------------------------------------
def optimize_query(sql: str, conn: sqlite3.Connection) -> str:
    """Add LIMIT if not present to prevent huge result sets"""
    upper_sql = sql.upper()
    if "LIMIT" not in upper_sql and "COUNT(" not in upper_sql and "SUM(" not in upper_sql and "AVG(" not in upper_sql:
        sql = f"{sql} LIMIT 1000"
    return sql

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="AI SQL Query App", layout="wide")
st.title("📊 Smart SQL Query Generator with AI")
st.markdown(" Upload your data and ask questions in natural language!. But make sure usage of strict prompting")


# Sidebar with tips
with st.sidebar:
    st.header("💡 Query Tips")
    st.markdown(
        """
**Example Questions:**
- Show all records for John Smith
- How many customers have "Premium" in their name?
- List top 5 products by price
- Find all orders that contain "laptop"
- What's the average age of users?
- Count customers by city
- Show me records where status is "Active"

**Keywords:**
- "has", "contains" → searches within text
- "top", "highest" → sorts descending
- "count", "how many" → counts records
- "average", "mean" → calculates average
"""
    )

uploaded_file = st.file_uploader("📁 Upload CSV or Excel", type=["csv", "xls", "xlsx"])
db_path = "./temp.db"

if uploaded_file:
    if st.button("🔄 Load File into Database"):
        filepath = save_uploaded_file(uploaded_file)
        with st.spinner("Loading and indexing data..."):
            try:
                conn = load_file_into_sqlite(filepath, db_path, TABLE_NAME)
                st.success("✅ File loaded successfully!")

                schema, schema_desc, columns, unique_values = get_enhanced_schema(conn, TABLE_NAME)

                with st.expander("📋 View Table Schema"):
                    st.code(schema, language="sql")

                with st.expander("📊 View Data Sample"):
                    try:
                        df_sample = pd.read_sql(f'SELECT * FROM "{TABLE_NAME}" LIMIT 10', conn)
                        st.dataframe(df_sample)
                        count_df = pd.read_sql(f'SELECT COUNT(*) as count FROM "{TABLE_NAME}"', conn)
                        total_rows = int(count_df["count"].iloc[0]) if not count_df.empty else 0
                        st.info(f"Total rows: {total_rows}")
                    except Exception:
                        st.info("Unable to preview data sample.")

                conn.close()
            except Exception as e:
                st.error(f"❌ Load error: {e}")

st.markdown("---")
question = st.text_input(
    "🤔 Ask your question in natural language:",
    placeholder="e.g., Show me all customers who have 'Gold' membership"
)

col1, col2 = st.columns([1, 5])
with col1:
    run_query = st.button("🚀 Run Query")
with col2:
    if st.button("🗑️ Clear Database"):
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                st.success("Database cleared!")
                st.experimental_rerun()
            except Exception:
                st.error("Unable to clear database file.")

if run_query:
    if not os.path.exists(db_path):
        st.error("⚠️ Please upload and load a file first.")
        st.stop()

    if not question.strip():
        st.error("⚠️ Please enter a question.")
        st.stop()

    conn = sqlite3.connect(db_path)
    try:
        schema, schema_desc, columns, unique_values = get_enhanced_schema(conn, TABLE_NAME)
    except Exception as e:
        st.error(f"Failed to read schema: {e}")
        conn.close()
        st.stop()

    with st.spinner("🤖 AI is generating SQL query..."):
        sql_query = generate_sql_query(question, schema, schema_desc, columns, conn, unique_values)

    if sql_query:
        st.subheader("🔍 Generated SQL Query")
        st.code(sql_query, language="sql")

        if not sql_query.lower().strip().startswith("select"):
            st.error("❌ Only SELECT queries are allowed for safety.")
            conn.close()
            st.stop()

        try:
            optimized_sql = optimize_query(sql_query, conn)

            cur = conn.cursor()
            cur.execute(optimized_sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []

            df = pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame(rows)

            st.subheader("📊 Query Results")
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                st.info(f"📌 Showing {len(df)} result rows")

                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No results found for your query.")

            with st.spinner("✨ Generating natural language answer..."):
                summary = generate_nl_summary(question, sql_query, rows, cols)

            st.subheader("💬 AI Answer")
            st.markdown(f"**{summary}**")

        except sqlite3.Error as e:
            st.error(f"❌ SQL Execution Error: {e}")
            st.info("💡 Try rephrasing your question or check if the column names are correct.")
        finally:
            conn.close()
    else:
        st.error("❌ Failed to generate SQL query. Please try rephrasing your question.")

# Footer
st.markdown("---")

