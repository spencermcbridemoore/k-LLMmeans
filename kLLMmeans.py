import numpy as np
import os
import anthropic
from openai import AzureOpenAI, OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, kmeans_plusplus, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids
#from InstructorEmbedding import INSTRUCTOR

from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file (if it exists)
load_dotenv(encoding="utf-8")

# Get API keys from environment variables (fallback to empty string if not set)
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
LLAMA_KEY = os.getenv("LLAMA_KEY", "")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY", "")
CLAUDE_KEY = os.getenv("CLAUDE_KEY", "")

AZURE_OPENAI_API_KEY = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
AZURE_OPENAI_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
AZURE_OPENAI_API_VERSION = (os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01").strip()
AZURE_EMBEDDING_DEPLOYMENT = (
    (os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or "").strip()
    or (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip()
)


def _normalize_azure_endpoint(raw: str) -> str:
    u = raw.rstrip("/")
    for suffix in ("/openai", "/openai/v1", "/v1"):
        if u.lower().endswith(suffix.lower()):
            u = u[: -len(suffix)].rstrip("/")
    return u


def _azure_embeddings_configured() -> bool:
    return bool(
        AZURE_OPENAI_API_KEY
        and AZURE_OPENAI_ENDPOINT
        and AZURE_EMBEDDING_DEPLOYMENT
    )


# Embeddings: Azure OpenAI (deployment name) if configured, else OpenAI API
_embeddings_backend = None  # "azure" | "openai" | None
if _azure_embeddings_configured():
    client_embeddings = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=_normalize_azure_endpoint(AZURE_OPENAI_ENDPOINT),
    )
    _embeddings_backend = "azure"
elif OPENAI_KEY:
    client_embeddings = OpenAI(api_key=OPENAI_KEY)
    _embeddings_backend = "openai"
else:
    client_embeddings = None
    _embeddings_backend = None

def llm_api (prompt, assistant_prompt, model = "llama3.3-70b", max_completion_tokens =1000):
    print(model)

    if model == "gpt-4o" or model == 'gpt-3.5-turbo':
        if not OPENAI_KEY:
            raise ValueError("OPENAI_KEY not set. Please set it in your .env file or environment variables.")
        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
        messages=[
                {"role": "system", "content": "You are a helpful assistant summarizing text clusters."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_prompt},
            ],
            max_tokens=1000,
            temperature=0,
            model=model
        )
        response_text = response.choices[0].message.content
        print(response_text)
        
    elif model == "llama3.3-70b":
        if not LLAMA_KEY:
            raise ValueError("LLAMA_KEY not set. Please set it in your .env file or environment variables.")
        client = OpenAI(
            api_key=LLAMA_KEY,
            base_url="https://api.llama-api.com/"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant summarizing text clusters."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_prompt},
            ],
            temperature=0,
            max_completion_tokens=max_completion_tokens,
            top_p=1,
            stream=False,
            stop=None,
            seed = 1
        )

        response_text = response.choices[0].message.content

    # Deepseek-chat
    elif model == "deepseek-chat":
        if not DEEPSEEK_KEY:
            raise ValueError("DEEPSEEK_KEY not set. Please set it in your .env file or environment variables.")
        client = OpenAI( 
            api_key=DEEPSEEK_KEY,
            base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant summarizing text clusters."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_prompt}
            ],
            stream=False,
            temperature=0,
            max_tokens=max_completion_tokens,
            top_p=1,
            stop=None,
            seed = 1
        )
        if response is None:
            response_text = ""
            print(response)
            # raise Exception("Response is None")
        else:
            response_text = response.choices[0].message.content


    elif model == "claude-3-7-sonnet-20250219":
        if not CLAUDE_KEY:
            raise ValueError("CLAUDE_KEY not set. Please set it in your .env file or environment variables.")
        client = anthropic.Anthropic(api_key=CLAUDE_KEY)

        message = client.messages.create(
            model=model,
            max_tokens=max_completion_tokens,
            temperature=0,
            system="You are a helpful assistant summarizing text clusters.",
            messages=[
                {"role": "user", "content":  prompt},
                {"role": "assistant", "content": assistant_prompt}
            ]
        )
        response_text = message.content[0].text


    else:
        response_text = "Model not found."




    return response_text
  
def get_embeddings(texts, model = "text-embedding-3-small", emb_type = 'openai', instructor_prompt = ""):
    """
    Get embeddings for emb_type='openai' via Azure OpenAI (if AZURE_* + deployment set)
    or OpenAI API (OPENAI_KEY). Azure uses your deployment name as the API model parameter.
    """

    def inner_get_embedding(text, api_model: str):

        if (len(text) == 0):
            return []

        try:

            result = client_embeddings.embeddings.create(
                model=api_model,
                input=text
            )
            return result.data

        except Exception:

            if (len(text) == 0) or (len(text) == 1):
                return []

            text_A = text[:int(len(text)/2)]
            text_B = text[int(len(text)/2):]

            result_A = inner_get_embedding(text_A, api_model)
            result_B = inner_get_embedding(text_B, api_model)

            return result_A + result_B

    if emb_type == 'openai':
        if client_embeddings is None or _embeddings_backend is None:
            raise ValueError(
                "OpenAI embeddings not configured. Set OPENAI_KEY for api.openai.com, or set "
                "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, and "
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT (Azure deployment name for text-embedding-3-small)."
            )
        api_model = AZURE_EMBEDDING_DEPLOYMENT if _embeddings_backend == "azure" else model
        result = inner_get_embedding(texts, api_model)
        embeddings = [i.embedding for i in result]
        return np.array(embeddings)
    
    elif emb_type == 'distilbert':
        model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
        embeddings = model.encode(texts)
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.array(normalized_embeddings)

    elif emb_type == 'e5-large':
        texts = ["query: " + text for text in texts]
        model = SentenceTransformer("intfloat/e5-large")
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)
    
    elif emb_type == 'sbert':
        model = SentenceTransformer("all-mpnet-base-v2")
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)
    
    elif emb_type == 'instructor':
        texts = [[instructor_prompt, text] for text in texts]
        model = INSTRUCTOR("hkunlp/instructor-xl")
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)

def _normalize_embeddings(embeddings, normalize=True):
    """
    Normalize embeddings to unit length (L2 normalization).
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Embeddings array of shape (n_samples, n_features)
    normalize : bool, default=True
        If True, normalize embeddings to unit length. If False, return original.
    
    Returns:
    --------
    np.ndarray
        Normalized embeddings if normalize=True, else original embeddings.
    """
    if not normalize:
        return embeddings
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms

def summarize_cluster(texts, prompt = "", text_type = "", model = "gpt-4o"):
    """
    Use an LLM to generate a summary of a cluster.
    """
    if prompt == "":
        prompt = f"Write a single sentence that represents the following cluster concisely:\n\n" + "\n".join(texts) 
    else:
        prompt = prompt + "\n\n" + "\n".join(texts)  
    
    if text_type == "":
        text_type = "Sentence:"

    messages = [
        {"role": "system", "content": "You are a helpful assistant summarizing text clusters."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text_type}
    ]
    
    response_text = llm_api(prompt,text_type,model)
    return response_text

def representative_cluster(texts,model="gpt-4o"):
    """
    Use an LLM to generate a summary of a cluster.
    """
    prompt = f"Identify the text that best represents the overall meaning and key themes of the following cluster of texts:\n\n" + "\n".join(texts)  # Limit to 5 for brevity
    
    text_type = "Representative Text:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant selecting the representatives of text clusters."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text_type}
    ]
    
    response_text = llm_api(prompt,text_type,model)
    return response_text

def sequentialMiniBatchKmeans(text_features, num_clusters, random_state, max_batch_size, max_iter = 100):

    num_batches = int(np.ceil(len(text_features)/max_batch_size))
    k, m = divmod(len(text_features), num_batches)

    #calculate batches
    text_features = [text_features[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_batches)]

    miniBatchKMeans = MiniBatchKMeans(n_clusters=num_clusters,
                         random_state=random_state,
                         batch_size=max_batch_size,
                         init="k-means++")
    
    #process each batch sequentially
    for cur_text_batch in text_features:
        miniBatchKMeans = miniBatchKMeans.partial_fit(cur_text_batch)
    
    return miniBatchKMeans

def _spherical_kmeans_assign_and_update(X, centroids, max_iter=1, random_state=None):
    """
    Perform spherical k-means assignment and centroid update.
    
    Spherical k-means clusters on the unit sphere using cosine similarity.
    Points are assigned to the centroid with highest cosine similarity (dot product),
    and centroids are updated as normalized sums of assigned points.
    
    Parameters:
    -----------
    X : np.ndarray
        Embeddings array of shape (n_samples, n_features). Should be normalized.
    centroids : np.ndarray
        Initial centroids of shape (n_clusters, n_features). Will be normalized.
    max_iter : int, default=1
        Maximum number of iterations for k-means updates.
    random_state : int or None, default=None
        Random state for reproducibility (currently unused, for API compatibility).
    
    Returns:
    --------
    assignments : np.ndarray
        Cluster assignments for each sample, shape (n_samples,)
    centroids : np.ndarray
        Updated centroids normalized to unit length, shape (n_clusters, n_features)
    """
    # Normalize centroids to unit length
    centroids = _normalize_embeddings(centroids, normalize=True)
    n_clusters = centroids.shape[0]
    
    for _ in range(max_iter):
        # Assignment: assign each point to centroid with highest cosine similarity
        # Cosine similarity = dot product when vectors are normalized
        similarities = X @ centroids.T  # Shape: (n_samples, n_clusters)
        assignments = np.argmax(similarities, axis=1)
        
        # Update centroids: for each cluster, sum assigned points and normalize
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            cluster_mask = assignments == k
            if np.any(cluster_mask):
                # Sum of assigned points, then normalize
                cluster_sum = np.sum(X[cluster_mask], axis=0)
                norm = np.linalg.norm(cluster_sum)
                if norm > 0:
                    new_centroids[k] = cluster_sum / norm
                else:
                    # If sum is zero vector, keep previous centroid
                    new_centroids[k] = centroids[k]
            else:
                # Empty cluster: keep previous centroid
                new_centroids[k] = centroids[k]
        
        # Check for convergence (centroids unchanged)
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return assignments, centroids

def spherical_kmeans_fit_predict(X, n_clusters, max_iter=300, random_state=None, normalize=True):
    """
    Perform spherical k-means clustering on embeddings.
    
    This is a vanilla (non-LLM) k-means algorithm that clusters on the unit sphere
    using cosine similarity instead of Euclidean distance. Points are assigned to
    the centroid with highest cosine similarity, and centroids are updated as
    normalized sums of assigned points.
    
    Parameters:
    -----------
    X : np.ndarray
        Embeddings array of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters to create.
    max_iter : int, default=300
        Maximum number of k-means iterations.
    random_state : int or None, default=None
        Random seed for k-means++ initialization and reproducibility.
    normalize : bool, default=True
        If True, normalize embeddings to unit length before clustering.
        Set to False if embeddings are already normalized.
    
    Returns:
    --------
    assignments : np.ndarray
        Cluster assignments for each sample, shape (n_samples,)
    centroids : np.ndarray
        Cluster centroids normalized to unit length, shape (n_clusters, n_features)
    
    Notes:
    ------
    - Centroids are always returned as unit vectors (L2 norm = 1), unlike
      sklearn's KMeans which returns unnormalized arithmetic means.
    - This matches the initialization behavior of sklearn's KMeans when using
      init='k-means++' (default).
    """
    # Validate inputs
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    if X.shape[0] < n_clusters:
        raise ValueError(f"n_samples ({X.shape[0]}) must be >= n_clusters ({n_clusters})")
    
    # Normalize embeddings if requested (work on copy to avoid mutating input)
    if normalize:
        X_work = _normalize_embeddings(X.copy(), normalize=True)
    else:
        X_work = X.copy()
    
    # Initialize centroids using k-means++ on normalized embeddings
    _, init_indices = kmeans_plusplus(X_work, n_clusters=n_clusters, random_state=random_state)
    init_centroids = X_work[init_indices]
    
    # Run spherical k-means
    assignments, centroids = _spherical_kmeans_assign_and_update(
        X_work, init_centroids, max_iter=max_iter, random_state=random_state)
    
    return assignments, centroids
    
def miniBatchKLLMeans(text_data, 
              num_clusters,
              max_batch_size = 5000, 
              init = 'k-means++',
              prompt = "", text_type = "",
              force_context_length = 0, max_llm_iter = 5, 
              max_iter = 100, tol=1e-4, random_state = None, 
              emb_type = 'openai', text_features = None,
              final_iter = True,
              initial_iter = True,
              model = "gpt-4o",
              geometry = "euclidean",
              normalize_embeddings = True):
    """
    Mini-batch version of kLLMmeans for processing large datasets in batches.
    
    Processes text data in batches sequentially, updating centroids across batches.
    Supports both Euclidean and spherical k-means geometries.
    
    Parameters:
    -----------
    text_data : list
        List of text strings to cluster.
    num_clusters : int
        Number of clusters to create.
    max_batch_size : int, default=5000
        Maximum number of samples per batch.
    init : str or np.ndarray, default='k-means++'
        Initialization method for centroids. For spherical geometry, array centroids will be normalized.
    prompt : str, default=""
        Prompt template for LLM summary generation.
    text_type : str, default=""
        Expected format of LLM response.
    force_context_length : int, default=0
        If > 0, subsample this many texts per cluster for summary generation.
    max_llm_iter : int, default=5
        Maximum number of LLM summary iterations per batch.
    max_iter : int, default=100
        Maximum number of k-means iterations per LLM iteration.
    tol : float, default=1e-4
        Convergence tolerance for centroid shift.
    random_state : int or None, default=None
        Random seed for reproducibility.
    emb_type : str, default='openai'
        Embedding type: 'openai', 'distilbert', 'e5-large', 'sbert', 'instructor'.
    text_features : np.ndarray or None, default=None
        Precomputed embeddings. If None, embeddings will be computed.
    final_iter : bool, default=True
        Whether to run final k-means iteration after LLM iterations.
    initial_iter : bool, default=True
        Whether to run initial k-means iteration before LLM iterations.
    model : str, default="gpt-4o"
        LLM model name for summary generation.
    geometry : str, default="euclidean"
        Clustering geometry: "euclidean" (standard k-means) or "spherical" (cosine similarity).
    normalize_embeddings : bool, default=True
        If True and geometry="spherical", normalize embeddings to unit length. Only applies to spherical mode.
    
    Returns:
    --------
    summaries : list
        List of summary lists (one per batch).
    centroids : np.ndarray
        Final merged centroids across all batches (normalized if geometry="spherical").
    """
    num_batches = int(np.ceil(len(text_data)/max_batch_size))
    k, m = divmod(len(text_data), num_batches)

    #calculate batches
    text_data = [text_data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_batches)]
    text_features = [text_features[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_batches)]

    #initialize centroids and no data processed
    centroids = init
    ndata = 0
    summaries = []

    #process each batch sequentially
    for ibatch in range(num_batches):
        
        batch_assignments, batch_summaries, _, batch_centroids, _, _ = kLLMmeans(text_data[ibatch], 
              num_clusters,
              init = centroids,
              prompt = prompt, text_type = text_type,
              force_context_length = force_context_length, max_llm_iter = max_llm_iter, 
              max_iter = max_iter, tol=tol, random_state = random_state, 
              emb_type = emb_type, text_features = text_features[ibatch],
              final_iter = final_iter,
              initial_iter = initial_iter,
              model = model,
              geometry = geometry,
              normalize_embeddings = normalize_embeddings)
        
        batch_counts = np.array([np.sum(np.array(batch_assignments)==i) for i in range(num_clusters)])

        if ibatch == 0:
            centroids = batch_centroids
            counts = batch_counts
        else:
            # Merge centroids based on geometry
            if geometry == "spherical":
                # For spherical: weighted sum then normalize
                # centroids and batch_centroids are already normalized
                weighted_sum = centroids * counts[:, None] + batch_centroids * batch_counts[:, None]
                centroids = _normalize_embeddings(weighted_sum, normalize=True)
            else:
                # For euclidean: weighted arithmetic mean
                centroids = np.array([(centroids[i]*counts[i] + batch_centroids[i]*batch_counts[i])/(counts[i] + batch_counts[i]) for i in range(num_clusters)])
            counts = counts + batch_counts

        summaries.append(batch_summaries)

    return summaries, centroids

def kLLMmeans(text_data, 
              num_clusters,
              init = 'k-means++',
              prompt = "", text_type = "",
              force_context_length = 0, max_llm_iter = 5, 
              max_iter = 100, tol=1e-4, random_state = None, 
              emb_type = 'openai', text_features = None,
              final_iter = True,
              initial_iter = True,
              instructor_prompt = "",
              model = "gpt-4o",
              geometry = "euclidean",
              normalize_embeddings = True):
    """
    Runs iterative KMeans clustering with dynamic centroid updates using LLM summaries.
    
    Parameters:
    -----------
    text_data : list
        List of text strings to cluster.
    num_clusters : int
        Number of clusters to create.
    init : str or np.ndarray, default='k-means++'
        Initialization method for centroids. Can be 'k-means++', 'random', or array of shape (n_clusters, n_features).
        For spherical geometry, array centroids will be normalized.
    prompt : str, default=""
        Prompt template for LLM summary generation.
    text_type : str, default=""
        Expected format of LLM response (e.g., "Sentence:", "Question:").
    force_context_length : int, default=0
        If > 0, subsample this many texts per cluster for summary generation.
    max_llm_iter : int, default=5
        Maximum number of LLM summary iterations.
    max_iter : int, default=100
        Maximum number of k-means iterations per LLM iteration.
    tol : float, default=1e-4
        Convergence tolerance for centroid shift.
    random_state : int or None, default=None
        Random seed for reproducibility.
    emb_type : str, default='openai'
        Embedding type: 'openai', 'distilbert', 'e5-large', 'sbert', 'instructor'.
    text_features : np.ndarray or None, default=None
        Precomputed embeddings. If None, embeddings will be computed.
    final_iter : bool, default=True
        Whether to run final k-means iteration after LLM iterations.
    initial_iter : bool, default=True
        Whether to run initial k-means iteration before LLM iterations.
    instructor_prompt : str, default=""
        Prompt for instructor embedding type.
    model : str, default="gpt-4o"
        LLM model name for summary generation.
    geometry : str, default="euclidean"
        Clustering geometry: "euclidean" (standard k-means) or "spherical" (cosine similarity).
    normalize_embeddings : bool, default=True
        If True and geometry="spherical", normalize embeddings to unit length. Only applies to spherical mode.
    
    Returns:
    --------
    cluster_assignments : np.ndarray
        Cluster assignments for each text.
    summaries : list
        Final LLM-generated summaries for each cluster.
    summary_embeddings : np.ndarray
        Embeddings of final summaries.
    cluster_centroids : np.ndarray
        Final cluster centroids (normalized if geometry="spherical").
    summaries_evolution : list
        List of summary lists for each iteration.
    centroids_evolution : list
        List of centroid arrays for each iteration.
    """
    summaries_evolution = []
    centroids_evolution = []

    if not isinstance(text_data, list):
        raise TypeError(f"Expected a list for variable text_data, but got {type(text_data).__name__}")

    if text_features is None:
        text_features = get_embeddings(text_data, emb_type = emb_type, instructor_prompt = instructor_prompt)
    
    # Normalize embeddings for spherical geometry if requested
    if geometry == "spherical" and normalize_embeddings:
        text_features = _normalize_embeddings(text_features, normalize=True)
    
    if (final_iter == False) and (initial_iter == False):
        tmp_kmeans_iterations = int(max_iter/(max_llm_iter-1))
    elif (final_iter == False) or (initial_iter == False):
        tmp_kmeans_iterations = int(max_iter/max_llm_iter)
    else: 
        tmp_kmeans_iterations = int(max_iter/(max_llm_iter+1))

    # Handle initial clustering based on geometry
    if geometry == "spherical":
        # For spherical k-means, handle init normalization
        if isinstance(init, np.ndarray):
            init_centroids = _normalize_embeddings(init.copy(), normalize=True)
        elif init == 'k-means++':
            # Use k-means++ initialization then normalize
            _, init_indices = kmeans_plusplus(text_features, n_clusters=num_clusters, random_state=random_state)
            init_centroids = _normalize_embeddings(text_features[init_indices], normalize=True)
        else:  # 'random' or other
            # Random initialization on sphere
            np.random.seed(random_state)
            init_centroids = np.random.randn(num_clusters, text_features.shape[1])
            init_centroids = _normalize_embeddings(init_centroids, normalize=True)
        
        if initial_iter == False:
            cluster_assignments, centroids = _spherical_kmeans_assign_and_update(
                text_features, init_centroids, max_iter=1, random_state=random_state)
        else:
            cluster_assignments, centroids = _spherical_kmeans_assign_and_update(
                text_features, init_centroids, max_iter=tmp_kmeans_iterations, random_state=random_state)
    else:  # euclidean
        if initial_iter == False:
            kmeans = KMeans(n_clusters=num_clusters, init=init, max_iter=1, random_state=random_state)
        else:
            kmeans = KMeans(n_clusters=num_clusters, init=init, max_iter=tmp_kmeans_iterations, random_state=random_state)
        cluster_assignments = kmeans.fit_predict(text_features)
        centroids = kmeans.cluster_centers_
    
    for iteration in tqdm(range(1, max_llm_iter + 1), desc="Iterating KMeans"):
        # Group texts by cluster
        clustered_texts = {i: [] for i in range(num_clusters)}
        
        # Select texts inside the cluster for summary generation
        if force_context_length > 0:
            clustered_indexes = {i: [] for i in range(num_clusters)}
            for i, cluster in zip(range(len(text_data)), cluster_assignments):
                clustered_indexes[cluster].append(i)
            
            for i_cluster, index_list in clustered_indexes.items():
                cur_cluster_embeddings = [text_features[i] for i in index_list]
                cur_cluster_texts = [text_data[i] for i in index_list] 
                
                if len(index_list)>force_context_length:
                    # For spherical mode, embeddings are already normalized, but kmeans_plusplus works in Euclidean space
                    # This is a heuristic subsampling, so we use it as-is
                    _, selected_indices = kmeans_plusplus(np.array(cur_cluster_embeddings), n_clusters=force_context_length, random_state=random_state)
                    clustered_texts[i_cluster] = [cur_cluster_texts[i] for i in selected_indices]
                else:
                    clustered_texts[i_cluster] = [cur_text for cur_text in cur_cluster_texts]
        
        else:
            for text, cluster in zip(text_data, cluster_assignments):
                clustered_texts[cluster].append(text)

        # Generate summaries for each cluster
        summaries = [summarize_cluster(clustered_texts[i], prompt, text_type, model) if clustered_texts[i] else "" for i in range(num_clusters)]
        summaries_evolution.append(summaries)

        # Obtain embeddings of summaries
        summary_embeddings = get_embeddings(summaries, emb_type = emb_type, instructor_prompt = instructor_prompt)
        
        # Normalize summary embeddings for spherical geometry if requested
        if geometry == "spherical" and normalize_embeddings:
            summary_embeddings = _normalize_embeddings(summary_embeddings, normalize=True)

        # Check for convergence (if centroid shift is small)
        if geometry == "spherical":
            # Use angular distance (1 - cosine similarity) for spherical geometry
            # Since both are normalized, cosine similarity = dot product
            cosine_similarities = np.sum(centroids * summary_embeddings, axis=1)
            angular_distances = 1 - cosine_similarities
            centroid_shift = np.sum(angular_distances)
        else:
            # Use L2 norm difference for Euclidean geometry
            centroid_shift = np.linalg.norm(centroids - summary_embeddings, axis=1).sum()
        
        if centroid_shift < tol:
            print(f"Converged after {iteration} iterations.")
            break

        # Update centroids with summary embeddings
        centroids = summary_embeddings
        centroids_evolution.append(centroids)

        # Refit clustering with updated centroids
        if geometry == "spherical":
            # Normalize centroids to ensure they're on the sphere
            centroids = _normalize_embeddings(centroids, normalize=True)
            if (final_iter == False) and (iteration == max_llm_iter):
                cluster_assignments, centroids = _spherical_kmeans_assign_and_update(
                    text_features, centroids, max_iter=1, random_state=random_state)
            else:
                cluster_assignments, centroids = _spherical_kmeans_assign_and_update(
                    text_features, centroids, max_iter=tmp_kmeans_iterations, random_state=random_state)
        else:  # euclidean
            if (final_iter == False) and (iteration == max_llm_iter):
                kmeans = KMeans(n_clusters=num_clusters, init=centroids, max_iter=1)
            else:
                kmeans = KMeans(n_clusters=num_clusters, init=centroids, max_iter=tmp_kmeans_iterations)
            
            # Assign data points to the nearest new centroids
            cluster_assignments = kmeans.fit_predict(text_features)

    # Get final centroids
    if geometry == "spherical":
        cluster_centroids = centroids  # Already normalized from last iteration
    else:
        cluster_centroids = kmeans.cluster_centers_

    return cluster_assignments, summaries, summary_embeddings, cluster_centroids, summaries_evolution, centroids_evolution

def kLLMmeansSpherical(text_data, 
              num_clusters,
              init = 'k-means++',
              prompt = "", text_type = "",
              force_context_length = 0, max_llm_iter = 5, 
              max_iter = 100, tol=1e-4, random_state = None, 
              emb_type = 'openai', text_features = None,
              final_iter = True,
              initial_iter = True,
              instructor_prompt = "",
              model = "gpt-4o"):
    """
    Convenience wrapper for kLLMmeans with spherical geometry.
    
    This function calls kLLMmeans with geometry="spherical" and normalize_embeddings=True.
    It provides a simpler interface for users who want spherical k-means clustering.
    
    Parameters:
    -----------
    Same as kLLMmeans(), except geometry and normalize_embeddings are fixed to "spherical" and True.
    
    Returns:
    --------
    Same as kLLMmeans().
    """
    return kLLMmeans(text_data,
              num_clusters=num_clusters,
              init=init,
              prompt=prompt,
              text_type=text_type,
              force_context_length=force_context_length,
              max_llm_iter=max_llm_iter,
              max_iter=max_iter,
              tol=tol,
              random_state=random_state,
              emb_type=emb_type,
              text_features=text_features,
              final_iter=final_iter,
              initial_iter=initial_iter,
              instructor_prompt=instructor_prompt,
              model=model,
              geometry="spherical",
              normalize_embeddings=True)


def kLLMmedoids(text_data, 
              num_clusters, 
              max_llm_iter = 5, 
              max_iter = 100, tol=1e-4, random_state = None, 
              emb_type = 'openai', labels = [], text_features = None,
              final_iter = False,
              initial_iter = False,
              model = "gpt-4o"):
    """
    Runs iterative KMeans clustering with dynamic centroid updates using LLM summaries.
    """
    
    if not isinstance(text_data, list):
        raise TypeError(f"Expected a list for variable text_data, but got {type(text_data).__name__}")

    if text_features is None:
        text_features = get_embeddings(text_data, emb_type = emb_type)
    
    if (final_iter == False) and (initial_iter == False):
        tmp_kmedoids_iterations = int(max_iter/(max_llm_iter-1))
    elif (final_iter == False) or (initial_iter == False):
        tmp_kmedoids_iterations = int(max_iter/(max_llm_iter+1))
    else: 
        tmp_kmedoids_iterations = int(max_iter/max_llm_iter)
    
    # Step 1: Run KMedioids to initialize centroids
    if initial_iter == False:
        kmedoids = KMedoids(n_clusters=num_clusters, max_iter = 1, random_state=random_state)
    else:
        kmedoids = KMedoids(n_clusters=num_clusters, max_iter = tmp_kmedoids_iterations, random_state=random_state)

    kmedoids.fit(text_features)
    cluster_assignments = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_
    medoids = text_features[medoid_indices]
    
    for iteration in tqdm(range(1, max_llm_iter + 1), desc="Iterating KMedoids"):
        # Group texts by cluster
        clustered_texts = {i: [] for i in range(num_clusters)}
        
        # Select texts inside the cluster for summary generation
        for text, cluster in zip(text_data, cluster_assignments):
            clustered_texts[cluster].append(text)

        # Generate summaries for each cluster
        representatives = [representative_cluster(clustered_texts[i], model) if clustered_texts[i] else "" for i in range(num_clusters)]
        
        # Obtain embeddings of summaries
        representative_embeddings = get_embeddings(representatives, emb_type = emb_type)

        # Check for convergence (if centroid shift is small)
        medoids_shift = np.linalg.norm(medoids - representative_embeddings, axis=1).sum()
        if medoids_shift < tol:
            print(f"Converged after {iteration} iterations.")
            break

        # Update centroids with summary embeddings
        medoids = representative_embeddings

        if (final_iter == False) and (iteration == max_llm_iter):
            kmedoids = KMedoids(n_clusters=num_clusters, init=medoids, max_iter=1)
        else:
            kmedoids = KMedoids(n_clusters=num_clusters, init=medoids, max_iter=tmp_kmedoids_iterations)
        
        # Assign data points to the nearest new centroids
        cluster_assignments = kmedoids.fit_predict(text_features)


    return cluster_assignments, representatives