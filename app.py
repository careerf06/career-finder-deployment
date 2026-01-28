from concurrent.futures import ThreadPoolExecutor, as_completed
import profile
import random
import sys
import threading
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from supabase import create_client, Client
import os   
from dotenv import load_dotenv
from flask_cors import CORS
import json
import re
import torch
import time
from datetime import datetime, timedelta
import hashlib
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from collections import Counter
import pickle
import platform


os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism

# Set PyTorch to use CPU only
torch.set_num_threads(1)  # Use single thread for PyTorch
torch.set_grad_enabled(False)  # Disable gradients for inference

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

nltk_setup_done = False

# ==================== PERSISTENT CACHE SYSTEM ====================
class PersistentCache:
    """Persistent cache system to prevent model re-downloads and speed up cold starts"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.getenv('PERSISTENT_CACHE_DIR', '/tmp/semantic_matching_cache')
        self.model_cache_dir = os.path.join(self.cache_dir, 'models')
        self.embedding_cache_dir = os.path.join(self.cache_dir, 'embeddings')
        
        # Create cache directories
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        
        # Set environment variables for HuggingFace cache
        os.environ['TRANSFORMERS_CACHE'] = self.model_cache_dir
        os.environ['HF_HOME'] = self.model_cache_dir
        os.environ['HF_HUB_CACHE'] = os.path.join(self.model_cache_dir, 'hub')
        
        logger.info(f"Persistent cache initialized at: {self.cache_dir}")
    
    def get_model_path(self, model_name: str) -> str:
        """Get model cache path"""
        safe_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.model_cache_dir, safe_name)
    
    def save_model_state(self, model_name: str, model_data: Any):
        """Save model state to disk"""
        model_path = self.get_model_path(model_name)
        try:
            with open(model_path + '.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model state saved to: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model state: {e}")
            return False
    
    def load_model_state(self, model_name: str) -> Optional[Any]:
        """Load model state from disk"""
        model_path = self.get_model_path(model_name) + '.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                logger.info(f"Model state loaded from: {model_path}")
                return model_data
            except Exception as e:
                logger.error(f"Failed to load model state: {e}")
        return None
    
    def save_embeddings(self, key: str, embeddings: np.ndarray):
        """Save embeddings to disk"""
        try:
            cache_path = os.path.join(self.embedding_cache_dir, hashlib.md5(key.encode()).hexdigest() + '.npz')
            np.savez_compressed(cache_path, embeddings=embeddings)
            return True
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return False
    
    def load_embeddings(self, key: str) -> Optional[np.ndarray]:
        """Load embeddings from disk"""
        try:
            cache_path = os.path.join(self.embedding_cache_dir, hashlib.md5(key.encode()).hexdigest() + '.npz')
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                return data['embeddings']
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
        return None

# Initialize persistent cache
persistent_cache = PersistentCache()
def convert_to_serializable(obj: Any) -> Any:
    """Convert non-serializable objects to JSON-serializable types"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

class LightweightSemanticEngine:
    """Lightweight semantic engine optimized for CPU-only Railway deployment"""
    
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        """
        Use bge-small-en-v1.5 instead of bge-large-en-v1.5
        - Small: 33M params (vs 335M in large)
        - 384 dimensions (vs 1024 in large)
        - Much faster CPU inference
        - Almost as good for semantic similarity
        """
        self.model_name = model_name
        self.model = None
        self.model_loaded = False
        self.model_lock = threading.Lock()
        
        # Lightweight NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        except (OSError, ImportError):
            logger.info("spaCy not available, using NLTK only")
            self.nlp = None
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Background model loading
        threading.Thread(target=self._load_model_background, daemon=True).start()
        
        logger.info(f"LightweightSemanticEngine initialized (will load {model_name} in background)")
    
    def _load_model_background(self):
        """Load model in background thread with persistence and CPU optimization"""
        try:
            model_cache_dir = persistent_cache.model_cache_dir
            
            logger.info(f"Starting background model loading: {self.model_name}")
            
            # Load model with CPU optimization
            with self.model_lock:
                start_time = time.time()
                
                # Try to load from persistent cache first
                model_state = persistent_cache.load_model_state(self.model_name)
                if model_state:
                    logger.info(f"Loading model from cache: {self.model_name}")
                    self.model = model_state
                else:
                    logger.info(f"Downloading model: {self.model_name}")
                    
                    # Use smaller batch size and simplified config for CPU
                    self.model = SentenceTransformer(
                        self.model_name,
                        device='cpu',
                        use_auth_token=False,
                        cache_folder=model_cache_dir
                    )
                    
                    # Save to persistent cache
                    persistent_cache.save_model_state(self.model_name, self.model)
                
                # Set to eval mode
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                
                # Freeze model for CPU optimization
                for param in self.model.parameters():
                    param.requires_grad = False
                
                self.model_loaded = True
                load_time = time.time() - start_time
                logger.info(f"✓ Model loaded in {load_time:.1f}s: {self.model_name}")
                
                # Memory cleanup
                import gc
                gc.collect()
                
        except Exception as e:
            logger.error(f"Failed to load model in background: {e}")
            self.model_loaded = False
    
    def ensure_model_loaded(self, timeout: int = 45) -> bool:
        """Ensure model is loaded before use, with timeout"""
        if self.model_loaded:
            return True
            
        logger.info(f"Waiting for model to load (timeout: {timeout}s)...")
        start_time = time.time()
        
        while not self.model_loaded:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"Model loading timeout after {timeout} seconds")
                return False
            
            # Log progress every 5 seconds
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                logger.info(f"Still loading model... ({int(elapsed)}s elapsed)")
            
            time.sleep(0.5)
        
        logger.info(f"Model loaded successfully after {time.time() - start_time:.1f}s")
        return True
    
    def get_semantic_embedding(self, text: str) -> np.ndarray:
        """Get semantic embedding for text with caching and CPU optimization"""
        if not text or not text.strip():
            return np.zeros(384, dtype=np.float32)
            
        # Try to get from persistent cache first
        cache_key = f"embedding_{hashlib.md5(text.encode()).hexdigest()}"
        cached_embedding = persistent_cache.load_embeddings(cache_key)
        
        if cached_embedding is not None:
            logger.debug(f"Cache hit for embedding: {cache_key[:20]}...")
            return cached_embedding
        
        # Ensure model is loaded
        if not self.ensure_model_loaded():
            logger.warning("Model not loaded, returning zeros")
            return np.zeros(384, dtype=np.float32)
        
        try:
            # Use smaller batch size for CPU
            with torch.no_grad():
                embedding = self.model.encode(
                    text, 
                    convert_to_tensor=False, 
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=8  # Smaller batch for CPU
                )
            
            embedding = embedding.astype(np.float32)
            
            # Save to persistent cache
            persistent_cache.save_embeddings(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def manual_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        try:
            if embedding1.size == 0 or embedding2.size == 0:
                return 0.0
            
            embedding1 = embedding1.flatten()
            embedding2 = embedding2.flatten()
            
            min_dim = min(embedding1.shape[0], embedding2.shape[0])
            
            dot_product = np.dot(embedding1[:min_dim], embedding2[:min_dim])
            norm1 = np.linalg.norm(embedding1[:min_dim])
            norm2 = np.linalg.norm(embedding2[:min_dim])
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(max(-1.0, min(1.0, similarity)))
            
        except Exception as e:
            logger.error(f"Error in cosine similarity: {e}")
            return 0.0
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return self.manual_cosine_similarity(embedding1, embedding2)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not text1.strip() or not text2.strip():
            return 0.0
            
        try:
            embedding1 = self.get_semantic_embedding(text1)
            embedding2 = self.get_semantic_embedding(text2)
            return self.calculate_cosine_similarity(embedding1, embedding2)
        except Exception as e:
            logger.error(f"Error in semantic similarity: {e}")
            return 0.0
    
    def calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """Calculate keyword-based similarity using Jaccard similarity"""
        try:
            tokens1 = self._preprocess_text(text1)
            tokens2 = self._preprocess_text(text2)
            
            if not tokens1 or not tokens2:
                return 0.0
            
            set1 = set(tokens1)
            set2 = set(tokens2)
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            if union == 0:
                return 0.0
                
            return float(intersection / union)
        except Exception as e:
            logger.error(f"Error in keyword similarity: {e}")
            return 0.0
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Fast text preprocessing"""
        if not text:
            return []
        
        try:
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            
            if self.nlp:
                doc = self.nlp(text)
                return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
            else:
                tokens = word_tokenize(text)
                return [self.lemmatizer.lemmatize(t) for t in tokens 
                       if t.lower() not in self.stop_words and t.isalpha()]
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return []
class STSConfig:
    """Configuration for Semantic Textual Similarity system"""
    
    def __init__(self):
        self.model_name = os.getenv('STS_MODEL_NAME', 'BAAI/bge-small-en-v1.5')
        self.similarity_threshold = float(os.getenv('STS_SIMILARITY_THRESHOLD', '0.5'))
        self.batch_size = int(os.getenv('STS_BATCH_SIZE', '16'))  
        self.max_workers = int(os.getenv('STS_MAX_WORKERS', '2'))  
        self.cache_ttl_minutes = int(os.getenv('STS_CACHE_TTL', '1440'))  
        self.embedding_dimension = 384 
        
        self.cosine_weight = 0.75
        self.skill_weight = 0.20
        self.experience_weight = 0.05
        
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

class STSMetrics:
    """Metrics collector for STS system"""
    
    def __init__(self):
        self.matching_times = []
        self.batch_sizes = []
        self.similarity_scores = []
        
    def record_matching_operation(self, duration: float, batch_size: int, matches_found: int):
        """Record metrics for matching operation"""
        self.matching_times.append(duration)
        self.batch_sizes.append(batch_size)
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.matching_times:
            return {}
            
        return {
            'avg_matching_time': np.mean(self.matching_times),
            'median_matching_time': np.median(self.matching_times),
            'max_matching_time': np.max(self.matching_times),
            'min_matching_time': np.min(self.matching_times),
            'total_operations': len(self.matching_times),
            'avg_batch_size': np.mean(self.batch_sizes)
        }

class JobApplicantMatcher:
    def __init__(self, supabase_url: str, supabase_key: str, model_name: str = 'BAAI/bge-small-en-v1.5'):
        """Initialize with lightweight semantic engine for Railway CPU"""
        try:
            if not supabase_url or not supabase_key:
                raise ValueError("Supabase URL and Key must be provided")
                
            self.supabase: Client = create_client(supabase_url, supabase_key)
            
            # Initialize lightweight engine
            self.semantic_engine = None
            self.model_name = model_name
            self.model_initialized = False
            self.init_lock = threading.Lock()
            
            # Initialize caches with persistent backing
            self.job_cache = {}
            self.profile_cache = {}
            self.embedding_cache = {}
            self.cache_ttl = timedelta(minutes=60)  # Longer TTL
            
            # CPU-optimized settings
            self.batch_size = 8  # Smaller batches for CPU
            self.max_workers = 2  # Fewer concurrent workers
            
            self.metrics = STSMetrics()
            
            # Start model initialization in background
            self._init_model_background()
            
            self.active_matching_locks = {}
            self.lock_manager = threading.Lock()
            
            logger.info(f"JobApplicantMatcher initialized (using lightweight model: {model_name})")
            
            self.test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize JobApplicantMatcher: {e}")
            raise
    
    def _init_model_background(self):
        """Initialize model in background thread"""
        def load_model():
            try:
                logger.info(f"Starting background initialization of lightweight model")
                with self.init_lock:
                    self.semantic_engine = LightweightSemanticEngine(self.model_name)
                    self.model_initialized = True
                    logger.info(f"✓ Lightweight semantic engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize semantic engine: {e}")
                self.model_initialized = False
        
        # Start in background thread
        threading.Thread(target=load_model, daemon=True).start()
    
    def ensure_engine_ready(self, timeout: int = 60) -> bool:
        """Ensure semantic engine is ready before use"""
        if self.model_initialized and self.semantic_engine:
            return True
            
        logger.info("Waiting for semantic engine to initialize...")
        start_time = time.time()
        
        while not (self.model_initialized and self.semantic_engine):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"Semantic engine initialization timeout after {timeout} seconds")
                return False
            
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                logger.info(f"Still initializing engine... ({int(elapsed)}s elapsed)")
            
            time.sleep(0.5)
        
        # Ensure model inside engine is loaded
        if self.semantic_engine:
            return self.semantic_engine.ensure_model_loaded(timeout=30)
        
        return False

    def test_connection(self):
        """Test database connection"""
        try:
            response = self.supabase.table('applicant_profiles').select('id', count='exact').limit(1).execute()
            logger.info("Connected to Supabase successfully")
        except Exception as e:
            logger.error(f"Connection test failed: {e}")

    def get_applicant_profile(self, user_id: str) -> Dict:
        """Fetch applicant profile with caching"""
        cache_key = f"applicant_{user_id}"
        now = datetime.now()
        
        if cache_key in self.profile_cache:
            data, timestamp = self.profile_cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return data
        
        try:
            response = self.supabase.table('applicant_profiles') \
                .select('*') \
                .eq('id', user_id) \
                .single() \
                .execute()
            
            if response.data:
                logger.info(f"Fetched applicant profile for user: {user_id}")
                self.profile_cache[cache_key] = (response.data, now)
                return response.data
            else:
                logger.warning(f"No applicant profile found for user: {user_id}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching applicant profile: {e}")
            return {}

    def get_all_applicant_profiles(self, limit: int = 1000) -> List[Dict]:
        """Fetch all applicant profiles for batch processing"""
        cache_key = f"all_applicants_{limit}"
        now = datetime.now()
        
        if cache_key in self.profile_cache:
            data, timestamp = self.profile_cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return data
        
        try:
            response = self.supabase.table('applicant_profiles') \
                .select('*') \
                .limit(limit) \
                .execute()
            
            logger.info(f"Fetched {len(response.data)} applicant profiles")
            self.profile_cache[cache_key] = (response.data, now)
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"Error fetching all applicant profiles: {e}")
            return []

    def get_job_profile(self, job_id: str) -> Dict:
        """Fetch job profile with caching"""
        cache_key = f"job_{job_id}"
        now = datetime.now()
        
        if cache_key in self.job_cache:
            data, timestamp = self.job_cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return data
        
        try:
            response = self.supabase.table('jobpost') \
                .select('*') \
                .eq('id', job_id) \
                .single() \
                .execute()
            
            if response.data:
                logger.info(f"Fetched job profile for job: {job_id}")
                self.job_cache[cache_key] = (response.data, now)
                return response.data
            else:
                logger.warning(f"No job profile found for job: {job_id}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching job profile: {e}")
            return {}

    def get_job_postings(self) -> List[Dict]:
        """Fetch active job postings with caching"""
        cache_key = "active_jobs"
        now = datetime.now()
        
        if cache_key in self.job_cache:
            data, timestamp = self.job_cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return data
        
        try:
            response = self.supabase.table('jobpost') \
                .select('*') \
                .eq('status', 'Open') \
                .eq('approval_status', 'approved') \
                .execute()
            logger.info(f"Fetched {len(response.data)} job postings")
            self.job_cache[cache_key] = (response.data, now)
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"Error fetching job postings: {e}")
            return []

    def get_all_job_postings(self, limit: int = 100) -> List[Dict]:
        """Fetch all active job postings for batch processing"""
        cache_key = f"all_jobs_{limit}"
        now = datetime.now()
        
        if cache_key in self.job_cache:
            data, timestamp = self.job_cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return data
        
        try:
            response = self.supabase.table('jobpost') \
                .select('*') \
                .eq('status', 'Open') \
                .eq('approval_status', 'approved') \
                .limit(limit) \
                .execute()
            
            logger.info(f"Fetched {len(response.data)} job postings")
            self.job_cache[cache_key] = (response.data, now)
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"Error fetching all job postings: {e}")
            return []

    def has_sufficient_profile_data(self, profile: Dict) -> Tuple[bool, str]:
        """Check if applicant profile has sufficient data for meaningful matching"""
        if not profile:
            return False, "No profile data found"
        
        description = (profile.get('description') or '').strip()
        has_description = len(description) > 50 
        
        skills = profile.get('skills') or []
        if isinstance(skills, str):
            try:
                skills = json.loads(skills) if skills.startswith('[') else [s.strip() for s in skills.split(',')]
            except:
                skills = [skills] if skills else []
        
        has_skills = len(skills) > 0 and any(len(str(skill).strip()) > 0 for skill in skills)
        
        position = (profile.get('position') or '').strip()
        company = (profile.get('company') or '').strip()
        has_context = len(position) > 0 or len(company) > 0
        
        if not has_description and not has_skills:
            return False, "Profile has no description and no skills listed"
        elif has_skills and not has_description and not has_context:
            return False, "Profile has skills but lacks description and context for meaningful matching"
        elif has_description and len(description) < 30:
            return False, "Profile description is too short for meaningful matching"
        else:
            return True, "Profile has sufficient data for matching"
    def calculate_description_similarity(self, job: Dict, profile: Dict) -> float:
        """
        Description similarity WITHOUT penalizing applicant for missing description
        """
        try:
            job_description = self._clean_text(job.get('description', '') or '')
            applicant_description = self._clean_text(profile.get('description', '') or '')
            
            # Job description missing → neutral
            if len(job_description) < 20:
                logger.debug("Job description too short - neutral score applied")
                return 0.0

            # Applicant description missing → DO NOT punish
            if len(applicant_description) < 20:
                logger.debug("Applicant description too short - no penalty applied")
                return 0.0

            description_similarity = self.semantic_engine.calculate_semantic_similarity(
                job_description, 
                applicant_description
            )

            logger.debug(f"Description similarity score: {description_similarity:.4f}")
            return float(min(description_similarity, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating description similarity: {e}")
            return 0.0

        
    def calculate_enhanced_weighted_score(self, cosine_score: float, skill_score: float, 
                             experience_score: float, description_score: float,
                             job: Dict = None, profile: Dict = None) -> Dict[str, float]:
       
        try:
            # Determine job complexity
            has_many_requirements = False
            if job:
                job_requirements = job.get('requirements', [])
                has_many_requirements = len(job_requirements) > 10
            
            # Check if experience score is valid
            has_valid_experience = (
                experience_score is not None and 
                experience_score > 0.01
            )
            
            # Adaptive weighting based on job type
            if has_valid_experience:
                if has_many_requirements:
                    # Skill-heavy job
                    weights = {
                        'cosine': 0.45,
                        'skill': 0.35,
                        'description': 0.15,
                        'experience': 0.05
                    }
                else:
                    # Experience matters more
                    weights = {
                        'cosine': 0.45,
                        'skill': 0.25,
                        'description': 0.15,
                        'experience': 0.15
                    }
                
                primary_score = (
                    weights['cosine'] * cosine_score +
                    weights['skill'] * skill_score +
                    weights['description'] * description_score +
                    weights['experience'] * experience_score
                )
                experience_score_output = float(max(0.0, min(1.0, experience_score)))
            else:
                # No experience data
                if has_many_requirements:
                    weights = {
                        'cosine': 0.60,
                        'skill': 0.25,
                        'description': 0.15
                    }
                else:
                    weights = {
                        'cosine': 0.60,
                        'skill': 0.25,
                        'description': 0.15
                    }
                
                primary_score = (
                    weights['cosine'] * cosine_score +
                    weights['skill'] * skill_score +
                    weights['description'] * description_score
                )
                experience_score_output = None
            
            def sigmoid_transform(score):
                """Spread scores better across 0-1 range"""
                return 1 / (1 + np.exp(-8 * (score - 0.5)))
            
            transformed_score = sigmoid_transform(primary_score)
            
            return {
                'similarity_score': float(max(0.0, transformed_score)),
                'cosine_score': float(max(0.0, cosine_score)),
                'skill_score': float(max(0.0, skill_score)),
                'description_score': float(max(0.0, description_score)),
                'experience_score': experience_score_output,
                'weights_used': weights,
                'raw_score': float(primary_score)
            }
        except Exception as e:
            logger.error(f"Error calculating enhanced weighted score: {e}")
            return {
                'similarity_score': 0.0,
                'cosine_score': 0.0,
                'skill_score': 0.0,
                'description_score': 0.0,
                'experience_score': None
            }
    def create_semantic_text_representation(self, profile: Dict, entity_type: str = "applicant", job: Dict = None) -> str:
        """
        Create semantic text representation using new skills and experience fields
        """
        try:
            if entity_type == "applicant":
                # Applicant side remains the same
                skills = profile.get('skills', []) or []
                if isinstance(skills, str):
                    skills = json.loads(skills) if skills.startswith('[') else skills.split(',')
                
                skills_clean = [str(s).strip().lower() for s in skills[:15] if s]
                description = str(profile.get('description', '') or '')[:500]
                
                return f"Description: {description} | Skills: {', '.join(skills_clean)}"
                
            else:  # job
                # Use dedicated skills field instead of requirements
                skills = profile.get('skills', []) or []
                if isinstance(skills, str):
                    skills = json.loads(skills) if skills.startswith('[') else skills.split(',')
                
                # Fallback to requirements if skills is empty
                if not skills:
                    skills = profile.get('requirements', []) or []
                    if isinstance(skills, str):
                        skills = json.loads(skills) if skills.startswith('[') else skills.split(',')
                
                skills_clean = [str(s).strip().lower() for s in skills[:12] if s]
                description = str(profile.get('description', '') or '')[:500]
                
                # Include experience requirement in representation
                experience_req = float(profile.get('experience_required', 0) or 0)
                exp_text = f"Experience: {experience_req} years" if experience_req > 0 else ""
                
                parts = [f"Description: {description}", f"Skills: {', '.join(skills_clean)}"]
                if exp_text:
                    parts.append(exp_text)
                
                return " | ".join(parts)
                
        except Exception as e:
            logger.error(f"Error creating text representation: {e}")
            return "No information available"

    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning for semantic analysis"""
        if not text:
            return ""
        try:
            cleaned = re.sub(r'[^\w\s\.\,\!]', '', text.strip())
            cleaned = re.sub(r'\s+', ' ', cleaned)
            return cleaned[:500]
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return ""

    def _normalize_skills(self, skills) -> List[str]:
        """Enhanced skill normalization with semantic grouping"""
        try:
            if isinstance(skills, str):
                try:
                    skills = json.loads(skills) if skills.startswith('[') else [s.strip() for s in skills.split(',')]
                except:
                    skills = [skills] if skills else []
            
            normalized = []
            seen = set()
            for skill in skills:
                if isinstance(skill, str):
                    clean_skill = skill.lower().strip()
                    clean_skill = re.sub(r'^(expert|proficient|skilled|experienced|basic)\s+in\s+', '', clean_skill)
                    
                    if clean_skill and clean_skill not in seen and len(clean_skill) > 1:
                        normalized.append(clean_skill)
                        seen.add(clean_skill)
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing skills: {e}")
            return []
    def _extract_job_skills(self, job: Dict) -> List[str]:
        """Extract skills from job posting with priority on new skills field"""
        # Prioritize dedicated skills field over requirements
        job_skills = job.get('skills', []) or []
        
        if job_skills:
            if isinstance(job_skills, str):
                job_skills = json.loads(job_skills) if job_skills.startswith('[') else job_skills.split(',')
            return self._normalize_skills(job_skills)
        
        # Fallback to requirements if skills field is empty
        requirements = job.get('requirements', []) or []
        return self._normalize_skills(requirements)

    def _infer_level_from_years(self, years: float) -> str:
        """Infer experience level from years required"""
        if years == 0:
            return 'entry'
        elif years <= 2:
            return 'junior'
        elif years <= 5:
            return 'mid'
        elif years <= 8:
            return 'senior'
        else:
            return 'expert'
    def batch_encode_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
            """Optimized batch encoding for CPU with persistent caching"""
            if not texts:
                return np.array([])
                
            cache_key = hashlib.md5("|".join(texts).encode()).hexdigest()
            now = datetime.now()
            
            # Check memory cache
            if cache_key in self.embedding_cache:
                data, timestamp = self.embedding_cache[cache_key]
                if now - timestamp < self.cache_ttl:
                    return data
            
            # Check persistent cache
            cached_embeddings = persistent_cache.load_embeddings(cache_key)
            if cached_embeddings is not None:
                self.embedding_cache[cache_key] = (cached_embeddings, now)
                return cached_embeddings
            
            # Ensure engine is ready
            if not self.ensure_engine_ready():
                logger.error("Semantic engine not ready for batch encoding")
                return np.array([])
            
            try:
                embeddings = []
                # Smaller batches for CPU
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    valid_texts = [t for t in batch_texts if t and t.strip()]
                    
                    if not valid_texts:
                        continue
                        
                    with torch.no_grad():
                        batch_embeddings = self.semantic_engine.model.encode(
                            valid_texts, 
                            convert_to_tensor=False,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            batch_size=min(4, len(valid_texts))  # Very small batch for CPU
                        )
                    embeddings.append(batch_embeddings)
                
                if embeddings:
                    result = np.vstack(embeddings)
                    # Cache in memory
                    self.embedding_cache[cache_key] = (result, now)
                    # Cache persistently
                    persistent_cache.save_embeddings(cache_key, result)
                    return result
                else:
                    return np.array([])
            except Exception as e:
                logger.error(f"Error in batch encoding: {e}")
                return np.array([])

    # ==================== NEW SKILL FILTERING METHODS ====================

    def filter_matching_skills(self, job_requirements: List[str], applicant_skills: List[str], 
                            similarity_threshold: float = 0.6) -> Tuple[List[str], List[str]]:
        """
        Filter applicant skills to only include those that match job requirements
        Returns: (matching_skills, non_matching_skills)
        """
        try:
            job_reqs_normalized = set(self._normalize_skills(job_requirements))
            applicant_skills_normalized = self._normalize_skills(applicant_skills)
            
            matching_skills = []
            non_matching_skills = []
            
            for applicant_skill in applicant_skills_normalized:
                # Check for exact match first
                if applicant_skill in job_reqs_normalized:
                    matching_skills.append(applicant_skill)
                    continue
                
                # Check for semantic similarity with each job requirement
                max_similarity = 0.0
                for job_req in job_reqs_normalized:
                    similarity = self.semantic_engine.calculate_semantic_similarity(
                        applicant_skill, job_req
                    )
                    max_similarity = max(max_similarity, similarity)
                
                if max_similarity >= similarity_threshold:
                    matching_skills.append(applicant_skill)
                else:
                    non_matching_skills.append(applicant_skill)
            
            logger.info(f"Filtered skills: {len(matching_skills)} matching, {len(non_matching_skills)} non-matching")
            return matching_skills, non_matching_skills
            
        except Exception as e:
            logger.error(f"Error filtering matching skills: {e}")
            return applicant_skills_normalized, []

    def create_filtered_semantic_text_representation(self, profile: Dict, job: Dict = None, 
                                                 entity_type: str = "applicant") -> str:
        """
        Create semantic text representation using only matching skills when job context is provided
        This ensures applicants are not penalized for having extra skills
        """
        try:
            if entity_type == "applicant" and job:
                # Filter skills to only include those matching job requirements
                all_skills = self._normalize_skills(profile.get('skills', []) or [])
                job_requirements = self._normalize_skills(job.get('requirements', []) or [])
                
                matching_skills, _ = self.filter_matching_skills(job_requirements, all_skills)
                
                parts = [
                    f"Description: {self._clean_text(profile.get('description', ''))}",
                    f"Relevant Skills: {', '.join(matching_skills[:15])}",  # Only matching skills
                ]
            else:
                # Use original method for job or when no job context
                return self.create_semantic_text_representation(profile, entity_type)
            
            valid_parts = [part for part in parts if ':' in part and part.split(':', 1)[1].strip()]
            return " | ".join(valid_parts) if valid_parts else "No information available"
            
        except Exception as e:
            logger.error(f"Error creating filtered semantic text: {e}")
            return self.create_semantic_text_representation(profile, entity_type)

    def calculate_semantic_skill_similarity(self, job_requirements: List[str], applicant_skills: List[str]) -> float:
        """
        Enhanced skill similarity - multi-level matching
        """
        if not job_requirements or not applicant_skills:
            return 0.0
        
        try:
            job_reqs = set(self._normalize_skills(job_requirements))
            applicant_skills_set = set(self._normalize_skills(applicant_skills))
            
            if not job_reqs:
                return 0.0
            
            # Level 1: Exact matches (worth 1.0 each)
            exact_matches = job_reqs.intersection(applicant_skills_set)
            exact_coverage = len(exact_matches) / len(job_reqs)
            
            # Level 2: Semantic matches for unmatched requirements
            remaining_reqs = job_reqs - exact_matches
            remaining_applicant_skills = applicant_skills_set - exact_matches
            
            semantic_match_scores = []
            for req in remaining_reqs:
                best_score = 0.0
                for skill in remaining_applicant_skills:
                    similarity = self.semantic_engine.calculate_semantic_similarity(req, skill)
                    if similarity > best_score:
                        best_score = similarity
                
                # Only count if similarity is above threshold
                if best_score >= 0.65:
                    semantic_match_scores.append(best_score)
            
            semantic_coverage = sum(semantic_match_scores) / len(job_reqs) if job_reqs else 0
            
            # Combined score with higher weight on exact matches
            combined_score = (0.65 * exact_coverage) + (0.35 * semantic_coverage)
            
            logger.debug(f"Skill matching: exact={exact_coverage:.2f}, semantic={semantic_coverage:.2f}, final={combined_score:.2f}")
            
            return float(min(combined_score, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating semantic skill similarity: {e}")
            return 0.0

    def calculate_comprehensive_cosine_similarity_with_skill_filtering(
        self, jobs: List[Dict], profile: Dict) -> List[Dict]:
        """
        Calculate cosine similarity using only matching skills in the embeddings
        Returns list of dicts with scores and skill breakdown
        """
        if not jobs or not profile:
            return []
        
        try:
            similarity_results = []
            
            for job in jobs:
                # Get filtered skills for this specific job
                all_applicant_skills = self._normalize_skills(profile.get('skills', []))
                job_requirements = self._normalize_skills(job.get('requirements', []))
                
                matching_skills, non_matching_skills = self.filter_matching_skills(
                    job_requirements, all_applicant_skills
                )
                
                # Create filtered profile representation with only matching skills
                filtered_profile = profile.copy()
                filtered_profile['skills'] = matching_skills
                
                # Generate embeddings using filtered profile
                job_text = self.create_semantic_text_representation(job, "job")
                profile_text = self.create_filtered_semantic_text_representation(
                    filtered_profile, job, "applicant"
                )
                
                job_embedding = self.semantic_engine.get_semantic_embedding(job_text)
                profile_embedding = self.semantic_engine.get_semantic_embedding(profile_text)
                
                cosine_score = self.semantic_engine.calculate_cosine_similarity(
                    job_embedding, profile_embedding
                )
                
                # Get detailed skill scores
                skill_scores = self.calculate_filtered_semantic_skill_similarity(
                    job_requirements, all_applicant_skills, include_non_matching=False
                )
                
                similarity_results.append({
                    'cosine_score': float(cosine_score),
                    'skill_breakdown': skill_scores,
                    'matching_skills': matching_skills,
                    'non_matching_skills': non_matching_skills
                })
            
            return similarity_results
            
        except Exception as e:
            logger.error(f"Error in filtered cosine similarity calculation: {e}")
            return [{'cosine_score': 0.0, 'skill_breakdown': {}} for _ in jobs]
    def get_skill_matching_details(self, job_requirements: List[str], applicant_skills: List[str]) -> Dict:
        """Get detailed breakdown of how applicant skills match job requirements"""
        job_reqs = set(self._normalize_skills(job_requirements))
        app_skills = set(self._normalize_skills(applicant_skills))
        
        matched = job_reqs.intersection(app_skills)
        unmatched_requirements = job_reqs - app_skills
        extra_skills = app_skills - job_reqs
        
        return {
            'matched_requirements': list(matched),
            'unmatched_requirements': list(unmatched_requirements),
            'extra_skills': list(extra_skills),
            'requirements_coverage': len(matched) / len(job_reqs) if job_reqs else 0,
            'total_requirements': len(job_reqs),
            'total_applicant_skills': len(app_skills)
        }
    def perform_filtered_matching_applicant_to_jobs(self, user_id: str, 
                                                threshold: float = 0.5,
                                                save_to_db: bool = True) -> Dict[str, Any]:
        """
        Enhanced matching that only considers matching skills in embeddings
        This is used when use_skill_filtering=true in the API request
        """
        start_time = time.time()
        
        try:
            jobs = self.get_job_postings()
            profile = self.get_applicant_profile(user_id)
            
            if not jobs:
                logger.warning("No jobs found for filtered matching")
                return {
                    'matches': [],
                    'insufficient_data': False,
                    'message': 'No active job postings available for matching',
                    'total_matches': 0
                }

            if not profile:
                logger.warning(f"No profile found for user {user_id}")
                return {
                    'matches': [],
                    'insufficient_data': True,
                    'message': 'No applicant profile found',
                    'total_matches': 0
                }

            has_sufficient_data, data_message = self.has_sufficient_profile_data(profile)
            
            if not has_sufficient_data:
                logger.warning(f"Applicant {user_id} has insufficient data: {data_message}")
                return {
                    'matches': [],
                    'insufficient_data': True,
                    'message': data_message,
                    'total_matches': 0
                }
            
            logger.info(f"Starting FILTERED matching for user {user_id} with {len(jobs)} jobs")
            
            # Get filtered similarity scores
            similarity_results = self.calculate_comprehensive_cosine_similarity_with_skill_filtering(
                jobs, profile
            )
            
            matches = []
            for job_idx, job in enumerate(jobs):
                if job_idx >= len(similarity_results):
                    logger.warning(f"Job index {job_idx} out of range for similarity results")
                    continue
                
                result = similarity_results[job_idx]
                cosine_score = result['cosine_score']
                skill_breakdown = result['skill_breakdown']
                
                if cosine_score < threshold:
                    continue
                
                # Use the filtered skill score
                skill_score = skill_breakdown.get('overall_score', 0.0)
                
                # Enhanced experience scoring with proper None handling
                experience_score = self.calculate_experience_similarity(job, profile)
                
                # Calculate weighted score
                scores = self.calculate_cosine_weighted_score(cosine_score, skill_score, experience_score)
                
                if scores['similarity_score'] >= threshold:
                    match_strength = self.get_cosine_match_strength(scores['similarity_score'])
                    
                    match_data = {
                        'job_id': job['id'],
                        'applicant_id': user_id,
                        'scores': scores,
                        'skill_breakdown': skill_breakdown,
                        'matching_skills': result['matching_skills'],
                        'non_matching_skills': result['non_matching_skills'],
                        'job_title': job.get('title', 'Unknown Title'),
                        'job_company': job.get('company_name', 'Unknown Company'),
                        'match_strength': match_strength,
                        'analysis': {
                            'cosine_interpretation': self.interpret_cosine_match(scores),
                            'key_strengths': self.identify_key_strengths(scores, job, profile),
                            'improvement_areas': self.identify_improvement_areas(scores, job, profile),
                            'skill_match_details': {
                                'matching_skills_count': len(result['matching_skills']),
                                'non_matching_skills_count': len(result['non_matching_skills']),
                                'coverage_percentage': skill_breakdown.get('coverage_percentage', 0),
                                'matching_skills': result['matching_skills'][:10],  # Top 10 for display
                            },
                            'experience_analysis': self._get_experience_analysis(job, profile)
                        }
                    }
                    matches.append(match_data)
                    
                    # Save each match immediately to Supabase
                    if save_to_db:
                        try:
                            self.save_single_match_to_db(match_data, user_id=user_id)
                            logger.info(f"Saved filtered match for job {job['id']} with score {scores['similarity_score']:.4f}")
                        except Exception as save_error:
                            logger.error(f"Failed to save filtered match for job {job['id']}: {save_error}")
            
            # Sort matches by similarity score
            sorted_matches = sorted(matches, key=lambda x: x['scores']['similarity_score'], reverse=True)
            
            processing_time = time.time() - start_time
            logger.info(f"Filtered matching completed in {processing_time:.2f}s. Found {len(sorted_matches)} matches.")
            
            return {
                'matches': sorted_matches,
                'insufficient_data': False,
                'message': f'Found {len(sorted_matches)} matches using skill filtering',
                'total_matches': len(sorted_matches),
                'processing_time': processing_time,
                'matching_method': 'filtered_skills_only'
            }
            
        except Exception as e:
            logger.error(f"Error in filtered matching: {e}")
            return {
                'matches': [],
                'insufficient_data': False,
                'message': f'Error during matching: {str(e)}',
                'total_matches': 0
            }
    def _get_experience_analysis(self, job: Dict, profile: Dict) -> Dict:
        """Get detailed experience analysis for the match"""
        try:
            job_exp = self._extract_experience_from_job(job)
            applicant_exp = self._extract_experience_from_profile(profile)
            
            # Check if data exists
            job_has_data = bool(job_exp.get('level') or job_exp.get('years', 0) > 0 or 
                            (job_exp.get('description', '').strip() and len(job_exp.get('description', '').strip()) > 20))
            applicant_has_data = bool(applicant_exp.get('level') or applicant_exp.get('years', 0) > 0 or
                                    (applicant_exp.get('description', '').strip() and len(applicant_exp.get('description', '').strip()) > 20))
            
            analysis = {
                'job_requirements': {
                    'level': job_exp.get('level', 'Not specified') if job_has_data else 'No data',
                    'years': job_exp.get('years', 'Not specified') if job_has_data else 'No data',
                    'has_requirements': job_has_data
                },
                'applicant_qualifications': {
                    'level': applicant_exp.get('level', 'Not specified') if applicant_has_data else 'No data',
                    'years': applicant_exp.get('years', 'Not specified') if applicant_has_data else 'No data',
                    'has_experience': applicant_has_data
                },
                'compatibility': 'Not evaluated - insufficient data'
            }
            
            # Only determine compatibility if BOTH have data
            if not job_has_data or not applicant_has_data:
                return analysis
            
            # Simple compatibility check
            job_level = (job_exp.get('level', '') or '').lower()
            applicant_level = (applicant_exp.get('level', '') or '').lower()
            
            level_mapping = {'entry': 1, 'junior': 1, 'mid': 2, 'senior': 3, 'expert': 4}
            job_score = level_mapping.get(job_level, 2)
            applicant_score = level_mapping.get(applicant_level, 2)
            
            if applicant_score >= job_score:
                analysis['compatibility'] = 'Meets or exceeds requirements'
            elif applicant_score >= job_score - 1:
                analysis['compatibility'] = 'Partially meets requirements'
            else:
                analysis['compatibility'] = 'Below requirements'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in experience analysis: {e}")
            return {
                'job_requirements': {'level': 'Error', 'years': 'Error', 'has_requirements': False},
                'applicant_qualifications': {'level': 'Error', 'years': 'Error', 'has_experience': False},
                'compatibility': 'Error in analysis'
            }

    def calculate_comprehensive_cosine_similarity_applicant_to_jobs(self, jobs: List[Dict], profile: Dict) -> List[float]:
        """Calculate comprehensive cosine similarity between jobs and profile - ONE JOB AT A TIME"""
        if not jobs or not profile:
            return []
            
        try:
            # Generate profile embedding once (reused for all jobs)
            profile_text = self.create_semantic_text_representation(profile, "applicant")
            logger.info(f"Generating profile embedding...")
            profile_embedding = self.semantic_engine.get_semantic_embedding(profile_text)
            
            if profile_embedding.size == 0:
                logger.warning("No profile embedding generated")
                return [0.0] * len(jobs)
            
            logger.info(f"Profile embedding generated. Starting matching for {len(jobs)} jobs...")
            
            similarity_scores = []
            
            # Process each job individually
            for idx, job in enumerate(jobs):
                job_num = idx + 1
                job_title = job.get('title', 'Unknown Title')
                
                try:
                    # Log START of processing this job
                    logger.info(f"Processing job {job_num}/{len(jobs)}: '{job_title}'")
                    
                    # Generate embedding for this specific job
                    job_text = self.create_semantic_text_representation(job, "job")
                    job_embedding = self.semantic_engine.get_semantic_embedding(job_text)
                    
                    if job_embedding.size == 0:
                        logger.warning(f"Job {job_num}/{len(jobs)}: No embedding generated - Score: 0.0000")
                        similarity_scores.append(0.0)
                        continue
                    
                    # Calculate similarity for this job
                    similarity = self.semantic_engine.calculate_cosine_similarity(
                        job_embedding, profile_embedding
                    )
                    similarity_scores.append(similarity)
                    
                    # Log COMPLETION of this job with score
                    logger.info(f"Completed job {job_num}/{len(jobs)}: '{job_title}' - Score: {similarity:.4f}")
                    
                except Exception as job_error:
                    logger.error(f"Error processing job {job_num}/{len(jobs)}: {job_error}")
                    similarity_scores.append(0.0)
            
            logger.info(f"All {len(jobs)} jobs processed. Similarity calculation complete.")
            
            return similarity_scores
            
        except Exception as e:
            logger.error(f"Error in comprehensive cosine similarity calculation: {e}")
            return [0.0] * len(jobs) if jobs else []

    def calculate_comprehensive_cosine_similarity_job_to_applicants(self, applicants: List[Dict], job: Dict) -> List[float]:
        """Calculate comprehensive cosine similarity between applicants and job - ONE APPLICANT AT A TIME"""
        if not applicants or not job:
            return []
            
        try:
            # Generate job embedding once (reused for all applicants)
            job_text = self.create_semantic_text_representation(job, "job")
            logger.info(f"Generating job embedding...")
            job_embedding = self.semantic_engine.get_semantic_embedding(job_text)
            
            if job_embedding.size == 0:
                logger.warning("No job embedding generated")
                return [0.0] * len(applicants)
            
            logger.info(f"Job embedding generated. Starting matching for {len(applicants)} applicants...")
            
            similarity_scores = []
            
            # Process each applicant individually
            for idx, applicant in enumerate(applicants):
                applicant_num = idx + 1
                applicant_name = f"{applicant.get('first_name', '')} {applicant.get('last_name', '')}".strip() or f"Applicant {applicant.get('id', 'Unknown')}"
                
                try:
                    # Log START of processing this applicant
                    logger.info(f"Processing applicant {applicant_num}/{len(applicants)}: '{applicant_name}'")
                    
                    # Generate embedding for this specific applicant
                    applicant_text = self.create_semantic_text_representation(applicant, "applicant")
                    applicant_embedding = self.semantic_engine.get_semantic_embedding(applicant_text)
                    
                    if applicant_embedding.size == 0:
                        logger.warning(f"Applicant {applicant_num}/{len(applicants)}: No embedding generated - Score: 0.0000")
                        similarity_scores.append(0.0)
                        continue
                    
                    # Calculate similarity for this applicant
                    similarity = self.semantic_engine.calculate_cosine_similarity(
                        applicant_embedding, job_embedding
                    )
                    similarity_scores.append(similarity)
                    
                    # Log COMPLETION of this applicant with score
                    logger.info(f"Completed applicant {applicant_num}/{len(applicants)}: '{applicant_name}' - Score: {similarity:.4f}")
                    
                except Exception as applicant_error:
                    logger.error(f"Error processing applicant {applicant_num}/{len(applicants)}: {applicant_error}")
                    similarity_scores.append(0.0)
            
            logger.info(f"All {len(applicants)} applicants processed. Similarity calculation complete.")
            
            return similarity_scores
            
        except Exception as e:
            logger.error(f"Error in comprehensive cosine similarity calculation: {e}")
            return [0.0] * len(applicants) if applicants else []

    def calculate_semantic_skill_similarity(self, job_requirements: List[str], applicant_skills: List[str]) -> float:
        """
        Calculate semantic skill similarity using cosine similarity - NO PENALTY for extra applicant skills
        Only measures how well the applicant covers job requirements
        """
        if not job_requirements or not applicant_skills:
            return 0.0
        
        try:
            job_reqs = set(self._normalize_skills(job_requirements))
            applicant_skills_set = set(self._normalize_skills(applicant_skills))
            
            if not job_reqs:
                return 0.0
            
            # Calculate coverage: what % of job requirements does the applicant have?
            # NO PENALTY for having additional skills beyond requirements
            intersection = len(job_reqs.intersection(applicant_skills_set))
            coverage = intersection / len(job_reqs) if job_reqs else 0
            
            # For semantic comparison, only compare against job requirements
            # Don't include extra applicant skills in the comparison
            job_skills_text = " ".join(list(job_reqs)[:20])
            
            # Only use applicant skills that are relevant to job requirements
            # This prevents dilution from irrelevant extra skills
            matching_applicant_skills = list(job_reqs.intersection(applicant_skills_set))[:20]
            
            # If no exact matches, use all applicant skills for semantic comparison
            # but weight coverage heavily
            if matching_applicant_skills:
                applicant_skills_text = " ".join(matching_applicant_skills)
            else:
                applicant_skills_text = " ".join(list(applicant_skills_set)[:20])
            
            cosine_similarity_score = self.semantic_engine.calculate_semantic_similarity(
                job_skills_text, applicant_skills_text
            )
            
            # Modified Jaccard: only based on coverage of requirements, not union
            # This removes penalty for extra skills
            jaccard_similarity = coverage  # Same as coverage since we only care about job reqs
            
            # Weighted combination emphasizing coverage of required skills
            # Higher weight on coverage ensures extra skills don't hurt the score
            combined_score = (0.5 * coverage) + (0.4 * cosine_similarity_score) + (0.1 * jaccard_similarity)
            
            return float(min(combined_score, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating semantic skill similarity: {e}")
            return 0.0

    def calculate_filtered_semantic_skill_similarity(self, job_requirements: List[str], 
                                                 applicant_skills: List[str],
                                                 include_non_matching: bool = False) -> Dict[str, float]:
        """
        Calculate skill similarity with detailed breakdown - NO PENALTY for extra skills
        
        Args:
            job_requirements: List of required skills for the job
            applicant_skills: List of skills the applicant has
            include_non_matching: Whether to include non-matching skills in the score calculation
        
        Returns:
            Dictionary with detailed skill matching scores
        """
        if not job_requirements or not applicant_skills:
            return {
                'overall_score': 0.0,
                'coverage_score': 0.0,
                'semantic_score': 0.0,
                'exact_match_score': 0.0,
                'coverage_percentage': 0.0,
                'exact_matches': 0,
                'semantic_matches': 0,
                'total_requirements': len(job_requirements) if job_requirements else 0
            }
        
        try:
            job_reqs_normalized = set(self._normalize_skills(job_requirements))
            applicant_skills_normalized = self._normalize_skills(applicant_skills)
            
            if not job_reqs_normalized:
                return {'overall_score': 0.0, 'coverage_percentage': 0.0}
            
            # Exact matches
            exact_matches = []
            for skill in applicant_skills_normalized:
                if skill in job_reqs_normalized:
                    exact_matches.append(skill)
            
            # Semantic matches (for skills that don't exactly match)
            semantic_matches = []
            remaining_job_reqs = job_reqs_normalized - set(exact_matches)
            remaining_applicant_skills = [s for s in applicant_skills_normalized if s not in exact_matches]
            
            for job_req in remaining_job_reqs:
                best_match_score = 0.0
                best_match_skill = None
                
                for applicant_skill in remaining_applicant_skills:
                    similarity = self.semantic_engine.calculate_semantic_similarity(
                        job_req, applicant_skill
                    )
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_skill = applicant_skill
                
                if best_match_score >= 0.6:  # Threshold for semantic match
                    semantic_matches.append({
                        'job_requirement': job_req,
                        'applicant_skill': best_match_skill,
                        'similarity': best_match_score
                    })
            
            # Calculate coverage based ONLY on job requirements (no penalty for extra skills)
            total_matched = len(exact_matches) + len(semantic_matches)
            coverage_score = total_matched / len(job_reqs_normalized)
            coverage_percentage = coverage_score * 100
            
            # Exact match score (what % of requirements are exactly matched)
            exact_match_score = len(exact_matches) / len(job_reqs_normalized)
            
            # Semantic match contribution
            semantic_contribution = sum([m['similarity'] for m in semantic_matches]) / len(job_reqs_normalized) if semantic_matches else 0
            
            # Overall score: heavily weighted towards covering requirements
            overall_score = (
                0.6 * exact_match_score +  # 60% weight on exact matches
                0.4 * semantic_contribution  # 40% weight on semantic matches
            )
            
            return {
                'overall_score': float(min(overall_score, 1.0)),
                'coverage_score': float(coverage_score),
                'coverage_percentage': float(coverage_percentage),
                'exact_match_score': float(exact_match_score),
                'semantic_score': float(semantic_contribution),
                'exact_matches': len(exact_matches),
                'semantic_matches': len(semantic_matches),
                'total_requirements': len(job_reqs_normalized),
                'total_matched': total_matched,
                'exact_match_details': exact_matches[:10],  # Top 10 for display
                'semantic_match_details': semantic_matches[:10]  # Top 10 for display
            }
            
        except Exception as e:
            logger.error(f"Error calculating filtered semantic skill similarity: {e}")
            return {'overall_score': 0.0, 'coverage_percentage': 0.0}


    def calculate_experience_similarity(self, job: Dict, profile: Dict) -> Optional[float]:
        """
        Calculate experience similarity using cosine similarity with proper None handling
        Returns None if there's insufficient experience data
        """
        try:
            job_experience = self._extract_experience_from_job(job)
            applicant_experience = self._extract_experience_from_profile(profile)
            
            # Check if BOTH job and applicant have meaningful experience data
            job_has_experience = (
                job_experience and 
                (job_experience.get('level') and job_experience.get('level', '').lower() not in ['', 'none', 'not required', 'any']) or
                (job_experience.get('years', 0) > 0) or
                (job_experience.get('description', '').strip() and len(job_experience.get('description', '').strip()) > 20)
            )
            
            applicant_has_experience = (
                applicant_experience and 
                (applicant_experience.get('level') and applicant_experience.get('level', '').lower() not in ['', 'none']) or
                (applicant_experience.get('years', 0) > 0) or
                (applicant_experience.get('description', '').strip() and len(applicant_experience.get('description', '').strip()) > 20)
            )
            
            # If EITHER has no experience data, return None (blank)
            if not job_has_experience or not applicant_has_experience:
                logger.debug("Insufficient experience data - returning None")
                return None
                
            # Both have experience data - calculate similarity
            experience_score = self._compare_experience_levels(job_experience, applicant_experience)
            
            # Safely calculate cosine similarity with proper None handling
            job_desc = job_experience.get('description', '') or ''
            applicant_desc = applicant_experience.get('description', '') or ''
            
            if not job_desc.strip() or not applicant_desc.strip():
                exp_cosine_score = 0.3
            else:
                exp_cosine_score = self.semantic_engine.calculate_semantic_similarity(job_desc, applicant_desc)
            
            combined_experience_score = (0.6 * experience_score) + (0.4 * exp_cosine_score)
            return float(min(combined_experience_score, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating experience similarity: {e}")
            return None  # Return None on error instead of 0.3
    def _job_requires_experience(self, job_experience: Dict) -> bool:
        """Determine if the job actually requires experience"""
        try:
            # Check if experience level indicates no experience required
            experience_level = job_experience.get('level', '').lower()
            no_experience_keywords = ['entry', 'junior', 'fresh', 'graduate', 'trainee', 'intern']
            
            # Check if years experience is 0 or not specified
            years_experience = job_experience.get('years', 0)
            
            # Check job description for no experience requirements
            description = job_experience.get('description', '').lower()
            no_exp_phrases = [
                'no experience required', 
                'no prior experience',
                'experience not required',
                'entry level',
                'fresh graduate',
                'will train',
                'training provided'
            ]
            
            # If job explicitly says no experience required, return False
            for phrase in no_exp_phrases:
                if phrase in description:
                    logger.info(f"Job explicitly states '{phrase}', experience not required")
                    return False
            
            # If experience level is entry-level and years is 0 or 1, consider as no experience required
            if any(keyword in experience_level for keyword in no_experience_keywords):
                if years_experience <= 1:
                    logger.info(f"Job is {experience_level} level with {years_experience} years experience - considered as no experience required")
                    return False
            
            # Default to requiring experience if none of the above conditions are met
            return True
            
        except Exception as e:
            logger.error(f"Error determining if job requires experience: {e}")
            return True  # Default to requiring experience for safety

    def _extract_experience_from_job(self, job: Dict) -> Dict:
        """Extract experience requirements from job posting with proper None handling"""
        try:
            description = job.get('description', '') or ''
            requirements = job.get('requirements', []) or []
            
            # Convert requirements to string safely
            requirements_text = ""
            if requirements:
                if isinstance(requirements, list):
                    requirements_text = " ".join([str(req) for req in requirements if req])
                else:
                    requirements_text = str(requirements)
            
            # Use the new experience_required field (double precision/float)
            years_required = float(job.get('experience_required', 0) or 0)
            
            experience_info = {
                'level': job.get('experience_level', '') or self._infer_level_from_years(years_required),
                'years': years_required,  # Now directly from experience_required field
                'description': f"{description} {requirements_text}".strip(),
                'type': job.get('job_type', '') or ''
            }
            return experience_info
        except Exception as e:
            logger.error(f"Error extracting experience from job: {e}")
            return {'level': '', 'years': 0, 'description': '', 'type': ''}
    def _extract_experience_from_profile(self, profile: Dict) -> Dict:
        """Extract experience information from applicant profile with proper None handling"""
        try:
            description = profile.get('description', '') or ''
            
            years = profile.get('experience', 0) or 0
            
            level = self._infer_level_from_years(years)
            
            experience_info = {
                'level': level,
                'years': years,
                'description': f"{description}".strip(),
            }
            return experience_info
        except Exception as e:
            logger.error(f"Error extracting experience from profile: {e}")
            return {'level': '', 'years': 0, 'description': ''}

    def _compare_experience_levels(self, job_exp: Dict, applicant_exp: Dict) -> float:
        """Compare experience levels and years with enhanced logic"""
        try:
            level_mapping = {
                'entry': 1, 'junior': 1, 'associate': 1,
                'mid': 2, 'intermediate': 2, 'medior': 2, 'mid-level': 2,
                'senior': 3, 'senior-level': 3, 'advanced': 3,
                'expert': 4, 'lead': 4, 'principal': 4, 'staff': 4, 'director': 5
            }
            
            job_level = (job_exp.get('level', '') or '').lower().strip()
            applicant_level = (applicant_exp.get('level', '') or '').lower().strip()
            
            if not job_level or job_level in ['', 'any', 'not specified', 'not required']:
                return 0.7  
            
            job_level_score = level_mapping.get(job_level, 2) 
            applicant_level_score = level_mapping.get(applicant_level, 2)
            
            level_diff = applicant_level_score - job_level_score
            
            if level_diff >= 0:
                level_score = 1.0
            else:
                level_score = max(0, 1.0 + (level_diff * 0.3)) 
            
            job_years = job_exp.get('years', 0) or 0
            applicant_years = applicant_exp.get('years', 0) or 0
            
            try:
                job_years = float(job_years) if job_years else 0
                applicant_years = float(applicant_years) if applicant_years else 0
            except (ValueError, TypeError):
                job_years = 0
                applicant_years = 0
            
            if job_years == 0:
                years_score = 0.8
            else:
                years_ratio = applicant_years / job_years if job_years > 0 else 1.0
                if years_ratio >= 1.0:
                    years_score = 1.0 
                elif years_ratio >= 0.5:
                    years_score = 0.7 
                else:
                    years_score = 0.3  
            
            experience_match = (0.5 * level_score) + (0.5 * years_score)
            return float(min(experience_match, 1.0))
            
        except Exception as e:
            logger.error(f"Error comparing experience levels: {e}")
            return 0.5 
    def calculate_cosine_weighted_score(self, cosine_score: float, skill_score: float, experience_score: float = 0.5) -> Dict[str, float]:
        try:
            primary_score = (0.75 * cosine_score) + (0.20 * skill_score) + (0.05 * experience_score)
            
            return {
                'similarity_score': float(max(0.0, primary_score)),
                'cosine_score': float(max(0.0, cosine_score)),
                'skill_score': float(max(0.0, skill_score)),
                'experience_score': float(max(0.0, experience_score))
            }
        except Exception as e:
            logger.error(f"Error calculating cosine weighted score: {e}")
            return {
                'similarity_score': 0.0,
                'cosine_score': 0.0,
                'skill_score': 0.0,
                'experience_score': 0.0
            }

    def get_existing_matches(self, user_id: str = None, job_id: str = None) -> Dict[str, Dict]:
        """Get existing matches for user or job to prevent duplicates"""
        try:
            
            if job_id and not user_id:
                query = self.supabase.table('applicant_match').select('*')
                query = query.eq('job_id', job_id)
            else:
                query = self.supabase.table('job_match_notification').select('*')
                if user_id:
                    query = query.eq('applicant_id', user_id)
                    
            response = query.execute()
            
            existing_matches = {}
            for match in response.data:
                key = f"{match['applicant_id']}_{match['job_id']}"
                existing_matches[key] = match
            return existing_matches
        except Exception as e:
            logger.error(f"Error fetching existing matches: {e}")
            return {}

    def _save_matches_individual(self, match_data: List[Dict], table_name: str):
        """Fallback method for individual match saving"""
        success_count = 0
        for match in match_data:
            try:
                result = self.supabase.table(table_name).upsert(match).execute()
                if not (hasattr(result, 'error') and result.error):
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to save individual match to {table_name}: {e}")
        
        logger.info(f"Individual fallback saved {success_count}/{len(match_data)} matches")

    def save_cosine_matches_to_db(self, matches: List[Dict], user_id: str = None, job_id: str = None):
        """Save cosine similarity matches with comprehensive scoring and better error handling"""
        if not matches:
            logger.info("No matches to save to database")
            return
            
        try:
            existing_matches = self.get_existing_matches(user_id, job_id)
            match_data = []
            current_time = datetime.now().isoformat()
            
            table_name = 'job_match_notification' if user_id else 'applicant_match'
            
            for match in matches:
                match_key = f"{match.get('applicant_id', user_id)}_{match.get('job_id', job_id)}"
                existing_match = existing_matches.get(match_key)
                
                score_changed = True
                if existing_match:
                    current_score = match['scores']['similarity_score']
                    previous_score = existing_match['similarity_score']
                    score_changed = abs(current_score - previous_score) > 0.01  
                
                if not existing_match or score_changed:
                    similarity_score = max(0.0, match['scores']['similarity_score'])
                    cosine_score = max(0.0, match['scores']['cosine_score'])
                    skill_score = max(0.0, match['scores']['skill_score'])
                    experience_score = max(0.0, match['scores'].get('experience_score', 0.0))
                    match_entry = {
                        'applicant_id': match.get('applicant_id', user_id),
                        'job_id': match.get('job_id', job_id),
                        'similarity_score': float(similarity_score),
                        'cosine_score': float(cosine_score),
                        'skill_score': float(skill_score),
                        'experience_score': float(experience_score),
                        'match_strength': match['match_strength'],
                        'updated_at': current_time
                    }
                    
                    if not existing_match:
                        match_entry['created_at'] = current_time
                    
                    match_data.append(match_entry)
            
            if match_data:
                try:
                    result = self.supabase.table(table_name).upsert(match_data).execute()
                    
                    if hasattr(result, 'error') and result.error:
                        logger.error(f"Database error saving to {table_name}: {result.error}")
                        self._save_matches_individual(match_data, table_name)
                    else:
                        logger.info(f"Successfully saved {len(match_data)} matches to {table_name}")
                        
                except Exception as db_error:
                    logger.error(f"Upsert failed for {table_name}: {db_error}")
                    self._save_matches_individual(match_data, table_name)
                    
            else:
                logger.info(f"No new or updated matches to save to {table_name}")
                
        except Exception as e:
            logger.error(f"Critical error saving matches: {e}")
            
    def perform_comprehensive_cosine_matching_applicant_to_jobs(
    self, user_id: str, threshold: float = 0.5, save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive cosine similarity matching for one applicant to all jobs.
        Modified with time-based lock to prevent only rapid duplicate clicks within cooldown period.
        Allows concurrent matching for different users and same user after cooldown.
        """
        
        COOLDOWN_SECONDS = 60  # Adjust this as needed
        
        with self.lock_manager:
            if user_id in self.active_matching_locks:
                last_request_time = self.active_matching_locks[user_id]
                time_since_last = time.time() - last_request_time
                
                if time_since_last < COOLDOWN_SECONDS:
                    remaining = int(COOLDOWN_SECONDS - time_since_last)
                    logger.warning(f"Rate limit hit for user {user_id}. {remaining}s remaining.")
                    return {
                        'matches': [],
                        'insufficient_data': False,
                        'message': f'Please wait {remaining} seconds before matching again',
                        'total_matches': 0,
                        'status': 'rate_limited',
                        'retry_after_seconds': remaining
                    }
            
            self.active_matching_locks[user_id] = time.time()
            logger.info(f"Lock registered for user {user_id}")

        try:
            start_time = time.time()

       
            logger.info("Checking if semantic engine is ready...")
            if not self.ensure_engine_ready(timeout=90):
                logger.error("Semantic engine not ready - cannot perform matching")
                return {
                    'matches': [],
                    'insufficient_data': False,
                    'message': 'Semantic matching engine is still initializing. Please try again in a few moments.',
                    'total_matches': 0,
                    'status': 'engine_not_ready'
                }
            
            if self.semantic_engine is None:
                logger.error("Semantic engine is None even after ensure_engine_ready")
                return {
                    'matches': [],
                    'insufficient_data': False,
                    'message': 'Semantic matching engine failed to initialize. Please contact support.',
                    'total_matches': 0,
                    'status': 'engine_initialization_failed'
                }
            
            logger.info("✓ Semantic engine is ready and loaded")

            jobs = self.get_job_postings()
            profile = self.get_applicant_profile(user_id)

            if not jobs:
                logger.warning("No jobs found for matching")
                return {
                    'matches': [],
                    'insufficient_data': False,
                    'message': 'No active job postings available for matching',
                    'total_matches': 0
                }

            if not profile:
                logger.warning(f"No profile found for user {user_id}")
                return {
                    'matches': [],
                    'insufficient_data': True,
                    'message': 'No applicant profile found',
                    'total_matches': 0
                }

       
            has_sufficient_data, data_message = self.has_sufficient_profile_data(profile)
            
            if not has_sufficient_data:
                logger.warning(f"Applicant {user_id} has insufficient data: {data_message}")
                return {
                    'matches': [],
                    'insufficient_data': True,
                    'message': data_message,
                    'total_matches': 0
                }

            logger.info(f"Starting ONE-BY-ONE matching for user {user_id} with {len(jobs)} jobs")

            logger.info(f"Generating profile embedding...")
            profile_start = time.time()
            profile_text = self.create_semantic_text_representation(profile, "applicant")
            
            if self.semantic_engine is None:
                logger.error("Semantic engine became None before profile embedding")
                return {
                    'matches': [],
                    'insufficient_data': False,
                    'message': 'Semantic engine error during processing',
                    'total_matches': 0
                }
            
            profile_embedding = self.semantic_engine.get_semantic_embedding(profile_text)
            logger.info(f"Profile embedding generated in {time.time() - profile_start:.2f}s")

            if profile_embedding.size == 0:
                logger.error("Failed to generate profile embedding")
                return {
                    'matches': [],
                    'insufficient_data': False,
                    'message': 'Failed to generate profile embedding',
                    'total_matches': 0
                }

            matches = []
            current_time = datetime.now().isoformat()


            logger.info(f"Processing {len(jobs)} jobs one at a time...")

            for job_idx, job in enumerate(jobs):
                job_num = job_idx + 1
                job_id = job['id']
                job_title = job.get('title', 'Unknown Title')

                logger.info(f"[{job_num}/{len(jobs)}] ===== Starting job: '{job_title}' =====")

                try:
                    if self.semantic_engine is None:
                        logger.error(f"[{job_num}/{len(jobs)}] Semantic engine became None - stopping")
                        break
                    
                    job_embedding_start = time.time()
                    job_text = self.create_semantic_text_representation(job, "job")
                    job_embedding = self.semantic_engine.get_semantic_embedding(job_text)
                    logger.info(f"[{job_num}/{len(jobs)}] Job embedding generated in "
                                f"{time.time() - job_embedding_start:.2f}s")

                    if job_embedding.size == 0:
                        logger.warning(f"[{job_num}/{len(jobs)}] No embedding — Skipping")
                        continue

                    cosine_score = self.semantic_engine.calculate_cosine_similarity(
                        job_embedding, profile_embedding
                    )
                    logger.info(f"[{job_num}/{len(jobs)}] Cosine score: {cosine_score:.4f}")

                    if cosine_score < threshold:
                        logger.info(f"[{job_num}/{len(jobs)}] Below threshold — Skipping")
                        continue

                    # Extract job skills using the new dedicated field
                    job_skills = self._extract_job_skills(job)
                    skill_score = self.calculate_semantic_skill_similarity(
                        job_skills,
                        profile.get('skills', [])
                    )
                    logger.debug(f"[{job_num}/{len(jobs)}] Skill score: {skill_score:.4f}")

                    experience_score = self.calculate_experience_similarity(job, profile)
                    if experience_score is not None:
                        logger.debug(f"[{job_num}/{len(jobs)}] Experience score: {experience_score:.4f}")
                    else:
                        logger.debug(f"[{job_num}/{len(jobs)}] Experience score: N/A")

                    description_score = self.calculate_description_similarity(job, profile)
                    logger.debug(f"[{job_num}/{len(jobs)}] Description score: {description_score:.4f}")

                    scores = self.calculate_enhanced_weighted_score(
                        cosine_score, skill_score, experience_score, description_score,
                        job=job, profile=profile
                    )
                    final_score = scores['similarity_score']
                    logger.info(f"[{job_num}/{len(jobs)}] Final weighted score: {final_score:.4f}")

                    if final_score < threshold:
                        logger.info(f"[{job_num}/{len(jobs)}] Final score below threshold — Skipping")
                        continue

                    match_strength = self.get_cosine_match_strength(final_score)
                    logger.info(f"[{job_num}/{len(jobs)}] Match strength: {match_strength}")

                    match_data = {
                        'job_id': job_id,
                        'applicant_id': user_id,
                        'scores': scores,
                        'job_title': job_title,
                        'job_company': job.get('company_name', 'Unknown Company'),
                        'match_strength': match_strength,
                        'analysis': {
                            'cosine_interpretation': self.interpret_enhanced_match(scores),
                            'key_strengths': self.identify_key_strengths_enhanced(scores, job, profile),
                            'improvement_areas': self.identify_improvement_areas(scores, job, profile),
                            'description_analysis': self._get_description_analysis(scores['description_score'])
                        }
                    }

                    matches.append(match_data)

                    if save_to_db:
                        try:
                            logger.info(f"[{job_num}/{len(jobs)}] Saving match to database...")
                            
                            match_entry = {
                                'applicant_id': user_id,
                                'job_id': job_id,
                                'similarity_score': float(scores['similarity_score']),
                                'cosine_score': float(scores['cosine_score']),
                                'skill_score': float(scores['skill_score']),
                                'description_score': float(scores.get('description_score', 0.0)),
                                'match_strength': match_strength,
                                'updated_at': current_time,
                                'created_at': current_time
                            }

                            if scores.get('experience_score') is not None:
                                match_entry['experience_score'] = float(scores['experience_score'])

                            result = self.supabase.table('job_match_notification').upsert(match_entry).execute()
                            
                            if hasattr(result, 'error') and result.error:
                                logger.error(f"[{job_num}/{len(jobs)}] ✗ Database error: {result.error}")
                            else:
                                logger.info(f"[{job_num}/{len(jobs)}] ✓ Match saved successfully")
                                
                        except Exception as save_error:
                            logger.error(f"[{job_num}/{len(jobs)}] ✗ Failed to save: {save_error}")

                    logger.info(f"[{job_num}/{len(jobs)}] ✓ COMPLETED: '{job_title}' - Score: {final_score:.4f}\n")

                except Exception as job_error:
                    logger.error(f"[{job_num}/{len(jobs)}] ✗ ERROR processing job: {job_error}\n")
                    continue

        
            logger.info(f"Sorting and calibrating {len(matches)} matches...")
            sorted_matches = sorted(matches, key=lambda x: x['scores']['similarity_score'], reverse=True)
            calibrated = self.calibrate_match_scores(sorted_matches)

            processing_time = time.time() - start_time
            logger.info(f"✓ Matching completed in {processing_time:.2f}s. Found {len(calibrated)} matches.")

            return {
                'matches': calibrated,
                'insufficient_data': False,
                'message': f'Found {len(calibrated)} matches with one-by-one processing',
                'total_matches': len(calibrated),
                'processing_time': processing_time,
                'scoring_method': 'one_by_one_individual_embeddings'
            }

        except Exception as e:
            logger.error(f"Error in one-by-one matching: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                'matches': [],
                'insufficient_data': False,
                'message': f'Error during matching: {str(e)}',
                'total_matches': 0
            }

        finally:
         
            with self.lock_manager:
                if len(self.active_matching_locks) > 100:
                    logger.info(f"Cleaning up old locks. Current count: {len(self.active_matching_locks)}")
                    
                    sorted_locks = sorted(
                        self.active_matching_locks.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    self.active_matching_locks = dict(sorted_locks[:100])
                    
                    logger.info(f"Locks cleaned up. New count: {len(self.active_matching_locks)}")

    def _get_description_analysis(self, description_score: float) -> str:
        """Provide interpretation of description matching"""
        if description_score >= 0.8:
            return "Excellent alignment between applicant's background and job description"
        elif description_score >= 0.6:
            return "Good match between applicant's experience and job requirements"
        elif description_score >= 0.4:
            return "Moderate alignment with job description"
        else:
            return "Limited description alignment - consider highlighting relevant experience"

    def interpret_enhanced_match(self, scores: Dict[str, float]) -> str:
        """Enhanced interpretation including description score"""
        similarity = scores['similarity_score']
        cosine = scores['cosine_score']
        skill = scores['skill_score']
        description = scores.get('description_score', 0)
        experience = scores.get('experience_score', 0)
        
        interpretations = []
        
        if cosine > 0.7:
            interpretations.append("strong overall semantic alignment")
        elif cosine > 0.5:
            interpretations.append("moderate overall alignment")
        
        if description > 0.7:
            interpretations.append("excellent description match")
        elif description > 0.5:
            interpretations.append("good description alignment")
            
        if skill > 0.7:
            interpretations.append("excellent skill match")
        elif skill > 0.5:
            interpretations.append("good skill overlap")
            
        if experience > 0.7:
            interpretations.append("strong experience match")
            
        return f"Match shows {', '.join(interpretations)}." if interpretations else "Basic match found."

    def identify_key_strengths_enhanced(self, scores: Dict[str, float], job: Dict, profile: Dict) -> List[str]:
        """Enhanced strength identification including description"""
        strengths = []
        
        if scores.get('description_score', 0) > 0.7:
            strengths.append("Strong alignment between background and job description")
        if scores['cosine_score'] > 0.7:
            strengths.append("Excellent overall semantic fit")
        if scores['skill_score'] > 0.7:
            strengths.append("High degree of skill compatibility")
        if scores.get('experience_score', 0) > 0.7:
            strengths.append("Excellent experience match")
            
        return strengths if strengths else ["Reasonable overall match"]
    
    def identify_key_strengths_enhanced(self, scores: Dict[str, float], job: Dict, profile: Dict) -> List[str]:
        """Enhanced strength identification including description"""
        strengths = []
        
        if scores.get('description_score', 0) > 0.7:
            strengths.append("Strong alignment between background and job description")
        if scores['cosine_score'] > 0.7:
            strengths.append("Excellent overall semantic fit")
        if scores['skill_score'] > 0.7:
            strengths.append("High degree of skill compatibility")
        if scores.get('experience_score', 0) > 0.7:
            strengths.append("Excellent experience match")
            
        return strengths if strengths else ["Reasonable overall match"]


    def save_enhanced_match_to_db(self, match: Dict, user_id: str = None, job_id: str = None):
        """Save match with description score to database"""
        try:
            existing_matches = self.get_existing_matches(user_id, job_id)
            current_time = datetime.now().isoformat()
            
            table_name = 'job_match_notification' if user_id else 'applicant_match'
            
            match_key = f"{match.get('applicant_id', user_id)}_{match.get('job_id', job_id)}"
            existing_match = existing_matches.get(match_key)
            
            score_changed = True
            if existing_match:
                current_score = match['scores']['similarity_score']
                previous_score = existing_match['similarity_score']
                score_changed = abs(current_score - previous_score) > 0.01
            
            if not existing_match or score_changed:
                # Get experience score and convert None to NULL for database
                experience_score = match['scores'].get('experience_score')
                
                match_entry = {
                    'applicant_id': match.get('applicant_id', user_id),
                    'job_id': match.get('job_id', job_id),
                    'similarity_score': float(match['scores']['similarity_score']),
                    'cosine_score': float(match['scores']['cosine_score']),
                    'skill_score': float(match['scores']['skill_score']),
                    'description_score': float(match['scores'].get('description_score', 0.0)),
                    'match_strength': match['match_strength'],
                    'updated_at': current_time
                }
                
                # Only add experience_score if it's not None
                if experience_score is not None:
                    match_entry['experience_score'] = float(experience_score)
                # If None, don't include the key (will be NULL in database)
                
                if not existing_match:
                    match_entry['created_at'] = current_time
                
                result = self.supabase.table(table_name).upsert(match_entry).execute()
                
                if not (hasattr(result, 'error') and result.error):
                    logger.debug(f"Successfully saved enhanced match to {table_name}")
                else:
                    logger.error(f"Error saving enhanced match: {result.error}")
                    
        except Exception as e:
            logger.error(f"Error saving enhanced match: {e}")


    def perform_comprehensive_cosine_matching_job_to_applicants(
    self, job_id: str, threshold: float = 0.5, save_to_db: bool = True
) -> Dict[str, Any]:
        """
        OPTIMIZED: Batch embeddings + Individual saves (same as applicant-to-jobs)
        """
        start_time = time.time()
        
        try:
            applicants = self.get_all_applicant_profiles()
            job = self.get_job_profile(job_id)
            
            if not applicants:
                logger.warning("No applicants found for cosine similarity matching")
                return {
                    'matches': [],
                    'insufficient_data': False,
                    'message': 'No applicant profiles available for matching',
                    'total_matches': 0
                }

            if not job:
                logger.warning(f"No job found for job_id {job_id}")
                return {
                    'matches': [],
                    'insufficient_data': True,
                    'message': 'No job posting found',
                    'total_matches': 0
                }

            # Filter valid applicants
            valid_applicants = []
            insufficient_data_applicants = []
            
            for applicant in applicants:
                has_sufficient_data, data_message = self.has_sufficient_profile_data(applicant)
                if has_sufficient_data:
                    valid_applicants.append(applicant)
                else:
                    insufficient_data_applicants.append({
                        'applicant_id': applicant['id'],
                        'reason': data_message
                    })
            
            if not valid_applicants:
                logger.warning(f"No applicants with sufficient data for job {job_id}")
                return {
                    'matches': [],
                    'insufficient_data': True,
                    'message': 'No applicants with sufficient profile data',
                    'total_matches': 0,
                    'insufficient_data_applicants': insufficient_data_applicants
                }
            
            logger.info(f"Starting OPTIMIZED matching for job {job_id} with {len(valid_applicants)} valid applicants")
            
            # ========================================
            # STEP 1: Generate job embedding ONCE
            # ========================================
            logger.info(f"Generating job embedding...")
            job_start = time.time()
            job_text = self.create_semantic_text_representation(job, "job")
            job_embedding = self.semantic_engine.get_semantic_embedding(job_text)
            logger.info(f"Job embedding generated in {time.time() - job_start:.2f}s")
            
            if job_embedding.size == 0:
                logger.error("Failed to generate job embedding")
                return {
                    'matches': [],
                    'insufficient_data': False,
                    'message': 'Failed to generate job embedding',
                    'total_matches': 0
                }
            
            # ========================================
            # STEP 2: Generate ALL applicant embeddings in ONE batch (KEY OPTIMIZATION!)
            # ========================================
            logger.info(f"Generating embeddings for ALL {len(valid_applicants)} applicants in batch...")
            batch_start = time.time()
            
            # Create all applicant text representations
            applicant_texts = []
            for applicant in valid_applicants:
                applicant_text = self.create_semantic_text_representation(applicant, "applicant")
                applicant_texts.append(applicant_text)
            
            # Generate ALL embeddings in ONE batch call - THIS IS THE SPEED FIX!
            try:
                with torch.no_grad():
                    applicant_embeddings_array = self.semantic_engine.model.encode(
                        applicant_texts,
                        convert_to_tensor=False,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=32  # Process 32 at a time internally
                    )
                
                batch_time = time.time() - batch_start
                logger.info(f"✓ ALL {len(valid_applicants)} applicant embeddings generated in {batch_time:.2f}s ({len(valid_applicants)/batch_time:.1f} applicants/sec)")
                
            except Exception as e:
                logger.error(f"Error generating applicant embeddings: {e}")
                return {
                    'matches': [],
                    'insufficient_data': False,
                    'message': f'Error generating embeddings: {str(e)}',
                    'total_matches': 0
                }
            
            # ========================================
            # STEP 3: Calculate ALL similarities using vectorization
            # ========================================
            logger.info(f"Calculating similarities for {len(valid_applicants)} applicants...")
            similarity_start = time.time()
            
            # Normalize job embedding
            job_norm = np.linalg.norm(job_embedding)
            if job_norm > 0:
                job_normalized = job_embedding / job_norm
            else:
                job_normalized = job_embedding
            
            # Normalize all applicant embeddings
            applicant_norms = np.linalg.norm(applicant_embeddings_array, axis=1, keepdims=True)
            applicant_norms[applicant_norms == 0] = 1
            applicant_embeddings_normalized = applicant_embeddings_array / applicant_norms
            
            # Calculate ALL cosine similarities at once (vectorized - super fast!)
            cosine_scores_array = np.dot(applicant_embeddings_normalized, job_normalized)
            
            logger.info(f"✓ All {len(valid_applicants)} similarities calculated in {time.time() - similarity_start:.2f}s")
            
            # ========================================
            # STEP 4: Process matches and save individually
            # ========================================
            logger.info(f"Processing and saving matches (threshold: {threshold})...")
            matches = []
            current_time = datetime.now().isoformat()
            
            for applicant_idx, applicant in enumerate(valid_applicants):
                applicant_num = applicant_idx + 1
                applicant_id = applicant['id']
                applicant_name = f"{applicant.get('first_name', '')} {applicant.get('last_name', '')}".strip() or f"Applicant {applicant_id}"
                
                # Get pre-calculated cosine score
                cosine_score = float(cosine_scores_array[applicant_idx])
                
                # Early exit if below threshold
                if cosine_score < threshold:
                    continue
                
                logger.info(f"[{applicant_num}/{len(valid_applicants)}] Processing: '{applicant_name}' (cosine: {cosine_score:.4f})")
                
                try:
                    # Calculate additional scores
                    # Extract job skills using the new dedicated field
                    job_skills = self._extract_job_skills(job)
                    skill_score = self.calculate_semantic_skill_similarity(
                        job_skills,
                        applicant.get('skills', [])  # ✅ FIXED: Changed from 'profile' to 'applicant'
                    )
                    logger.debug(f"[{applicant_num}/{len(valid_applicants)}] Skill score: {skill_score:.4f}")
                    
                    experience_score = self.calculate_experience_similarity(job, applicant)
                    if experience_score is not None:
                        logger.debug(f"[{applicant_num}/{len(valid_applicants)}] Experience score: {experience_score:.4f}")
                    else:
                        logger.debug(f"[{applicant_num}/{len(valid_applicants)}] Experience score: N/A")
                    
                    description_score = self.calculate_description_similarity(job, applicant)
                    logger.debug(f"[{applicant_num}/{len(valid_applicants)}] Description score: {description_score:.4f}")
                    # Calculate weighted score
                    scores = self.calculate_enhanced_weighted_score(
                        cosine_score, skill_score, experience_score, description_score,
                        job=job, profile=applicant
                    )
                    
                    final_score = scores['similarity_score']
                    logger.info(f"[{applicant_num}/{len(valid_applicants)}] Final weighted score: {final_score:.4f}")
                    
                    # Check final threshold
                    if final_score < threshold:
                        logger.info(f"[{applicant_num}/{len(valid_applicants)}] Final score below threshold - Skipping")
                        continue
                    
                    # Determine match strength
                    match_strength = self.get_cosine_match_strength(final_score)
                    logger.info(f"[{applicant_num}/{len(valid_applicants)}] Match strength: {match_strength}")
                    
                    # Build match data
                    match_data = {
                        'applicant_id': applicant_id,
                        'job_id': job_id,
                        'scores': scores,
                        'applicant_name': applicant_name,
                        'applicant_position': applicant.get('position', ''),
                        'match_strength': match_strength,
                        'analysis': {
                            'cosine_interpretation': self.interpret_enhanced_match(scores),
                            'key_strengths': self.identify_key_strengths_enhanced(scores, job, applicant),
                            'improvement_areas': self.identify_improvement_areas(scores, job, applicant),
                            'description_analysis': self._get_description_analysis(scores['description_score'])
                        }
                    }
                    
                    matches.append(match_data)
                    
                    # ========================================
                    # SAVE INDIVIDUALLY (same as applicant-to-jobs)
                    # ========================================
                    if save_to_db:
                        try:
                            logger.info(f"[{applicant_num}/{len(valid_applicants)}] Saving match to database...")
                            
                            experience_score_val = scores.get('experience_score')
                            match_entry = {
                                'applicant_id': applicant_id,
                                'job_id': job_id,
                                'similarity_score': float(scores['similarity_score']),
                                'cosine_score': float(scores['cosine_score']),
                                'skill_score': float(scores['skill_score']),
                                'description_score': float(scores.get('description_score', 0.0)),
                                'match_strength': match_strength,
                                'updated_at': current_time,
                                'created_at': current_time
                            }
                            
                            if experience_score_val is not None:
                                match_entry['experience_score'] = float(experience_score_val)
                            
                            # Individual save
                            result = self.supabase.table('applicant_match').upsert(match_entry).execute()
                            
                            if hasattr(result, 'error') and result.error:
                                logger.error(f"[{applicant_num}/{len(valid_applicants)}] ✗ Database error: {result.error}")
                            else:
                                logger.info(f"[{applicant_num}/{len(valid_applicants)}] ✓ Match saved successfully")
                                
                        except Exception as save_error:
                            logger.error(f"[{applicant_num}/{len(valid_applicants)}] ✗ Failed to save: {save_error}")
                    
                    logger.info(f"[{applicant_num}/{len(valid_applicants)}] ✓ COMPLETED: '{applicant_name}' - Score: {final_score:.4f}\n")
                    
                except Exception as applicant_error:
                    logger.error(f"[{applicant_num}/{len(valid_applicants)}] ✗ ERROR: {applicant_error}\n")
                    continue
            
            # ========================================
            # STEP 5: Sort and calibrate matches
            # ========================================
            logger.info(f"Sorting and calibrating {len(matches)} matches...")
            sorted_matches = sorted(matches, key=lambda x: x['scores']['similarity_score'], reverse=True)
            calibrated_matches = self.calibrate_match_scores(sorted_matches)
            
            processing_time = time.time() - start_time
            logger.info(f"✓ OPTIMIZED matching completed in {processing_time:.2f}s. Found {len(calibrated_matches)} matches.")
            
            return {
                'matches': calibrated_matches,
                'insufficient_data': False,
                'message': f'Found {len(calibrated_matches)} matches out of {len(valid_applicants)} valid applicants',
                'total_matches': len(calibrated_matches),
                'total_applicants_processed': len(applicants),
                'valid_applicants': len(valid_applicants),
                'insufficient_data_applicants': insufficient_data_applicants,
                'processing_time': processing_time,
                'scoring_method': 'optimized_batch_embeddings_individual_saves'
            }
                    
        except Exception as e:
            logger.error(f"Error in optimized job-to-applicants matching: {e}")
            return {
                'matches': [],
                'insufficient_data': False,
                'message': f'Error during matching: {str(e)}',
                'total_matches': 0
            }
    def save_single_match_to_db(self, match: Dict, user_id: str = None, job_id: str = None):
        """Save a single match to Supabase immediately"""
        try:
            existing_matches = self.get_existing_matches(user_id, job_id)
            current_time = datetime.now().isoformat()
            
            table_name = 'job_match_notification' if user_id else 'applicant_match'
            
            match_key = f"{match.get('applicant_id', user_id)}_{match.get('job_id', job_id)}"
            existing_match = existing_matches.get(match_key)
            
            score_changed = True
            if existing_match:
                current_score = match['scores']['similarity_score']
                previous_score = existing_match['similarity_score']
                score_changed = abs(current_score - previous_score) > 0.01  
            
            if not existing_match or score_changed:
                similarity_score = max(0.0, re.match['scores']['similarity_score'])
                cosine_score = max(0.0, min(1.0, match['scores']['cosine_score']))
                skill_score = max(0.0, min(1.0, match['scores']['skill_score']))
                experience_score_raw = match['scores'].get('experience_score')
                
                match_entry = {
                    'applicant_id': match.get('applicant_id', user_id),
                    'job_id': match.get('job_id', job_id),
                    'similarity_score': float(similarity_score),
                    'cosine_score': float(cosine_score),
                    'skill_score': float(skill_score),
                    'match_strength': match['match_strength'],
                    'updated_at': current_time
                }
                
                # Only add experience_score if it's not None
                if experience_score_raw is not None:
                    match_entry['experience_score'] = float(max(0.0, experience_score_raw))
                # If None, don't include the key (will be NULL in database)
                
                if not existing_match:
                    match_entry['created_at'] = current_time
                
                # Save individual match immediately
                try:
                    result = self.supabase.table(table_name).upsert(match_entry).execute()
                    
                    if hasattr(result, 'error') and result.error:
                        logger.error(f"Database error saving single match to {table_name}: {result.error}")
                    else:
                        logger.debug(f"Successfully saved single match to {table_name}")
                        
                except Exception as db_error:
                    logger.error(f"Upsert failed for single match in {table_name}: {db_error}")
                    
        except Exception as e:
            logger.error(f"Critical error saving single match: {e}")
    def get_cosine_match_strength(self, score: float) -> str:
        """Determine cosine similarity match strength"""
        try:
            percentage = score * 100
            
            if percentage >= 85:
                return 'Strong'
            elif percentage >= 75:
                return 'Good'
            elif percentage >= 65:
                return 'Moderate'
            elif percentage >= 45:
                return 'Fair'
            else:
                return 'Weak'
        except Exception as e:
            logger.error(f"Error determining match strength: {e}")
            return 'Unknown'

    def interpret_cosine_match(self, scores: Dict[str, float]) -> str:
        """Provide interpretation of the cosine similarity match scores"""
        try:
            similarity = scores['similarity_score']
            cosine = scores['cosine_score']
            skill = scores['skill_score']
            experience = scores.get('experience_score', 0)
            
            interpretations = []
            
            if cosine > 0.7:
                interpretations.append("strong cosine similarity alignment")
            elif cosine > 0.5:
                interpretations.append("moderate cosine similarity alignment")
            else:
                interpretations.append("limited cosine similarity alignment")
                
            if skill > 0.7:
                interpretations.append("excellent skill match")
            elif skill > 0.7:
                interpretations.append("good skill overlap")
                
            if experience > 0.7:
                interpretations.append("strong experience match")
            elif experience > 0.5:
                interpretations.append("relevant experience")
                
            return f"Match shows {', '.join(interpretations)}." if interpretations else "Basic match found."
        except Exception as e:
            logger.error(f"Error interpreting cosine match: {e}")
            return "Error in match interpretation"

    def identify_key_strengths(self, scores: Dict[str, float], job: Dict, profile: Dict) -> List[str]:
        """Identify key strengths of the cosine similarity match"""
        try:
            strengths = []
            
            if scores['cosine_score'] > 0.7:
                strengths.append("Strong contextual and cosine similarity alignment")
            if scores['skill_score'] > 0.7:
                strengths.append("High degree of skill compatibility")
            if scores.get('experience_score', 0) > 0.7:
                strengths.append("Excellent experience match")
            if scores['cosine_score'] > scores['skill_score']:
                strengths.append("Excellent overall contextual fit")
                
            return strengths if strengths else ["Reasonable overall match"]
        except Exception as e:
            logger.error(f"Error identifying key strengths: {e}")
            return ["Match identified"]

    def identify_improvement_areas(self, scores: Dict[str, float], job: Dict, profile: Dict) -> List[str]:
        """Identify areas for improvement in the cosine similarity match"""
        try:
            improvements = []
            
            if scores['skill_score'] < 0.5:
                improvements.append("Consider developing additional required skills")
            if scores['cosine_score'] < 0.5:
                improvements.append("Focus on building relevant domain knowledge")
            if scores.get('experience_score', 0) < 0.5:
                improvements.append("Gain more relevant experience in this field")
                
            return improvements
        except Exception as e:
            logger.error(f"Error identifying improvement areas: {e}")
            return []

    def get_top_cosine_matches(self, matches: List[Dict], top_n: int = 10) -> List[Dict]:
        """Get top N cosine similarity matches by primary score"""
        return matches[:top_n]
    
    def calibrate_match_scores(self, matches: List[Dict]) -> List[Dict]:
        """
        Calibrate scores to better use the 0-1 range
        """
        if not matches or len(matches) < 3:
            return matches
        
        try:
            scores = np.array([m['scores']['similarity_score'] for m in matches])
            
            # Temperature scaling for better distribution
            temperature = 1.5
            calibrated_scores = 1 / (1 + np.exp(-temperature * (scores - 0.5)))
            
            # Quantile normalization
            if len(calibrated_scores) > 1:
                ranks = np.argsort(np.argsort(calibrated_scores))
                normalized_scores = ranks / (len(ranks) - 1)
            else:
                normalized_scores = calibrated_scores
            
            # Update matches
            for i, match in enumerate(matches):
                original_score = match['scores']['similarity_score']
                match['scores']['similarity_score_uncalibrated'] = original_score
                match['scores']['similarity_score'] = float(normalized_scores[i])
                
                # Re-evaluate match strength with calibrated score
                match['match_strength'] = self.get_cosine_match_strength(normalized_scores[i])
            
            logger.info(f"Calibrated {len(matches)} match scores")
            return matches
            
        except Exception as e:
            logger.error(f"Error calibrating scores: {e}")
            return matches

    def process_single_applicant_matching(self, applicant_id: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Process cosine similarity matching for a single applicant"""
        try:
            logger.info(f"Processing cosine similarity matching for applicant: {applicant_id}")
            result = self.perform_comprehensive_cosine_matching_applicant_to_jobs(
                applicant_id, 
                threshold, 
                save_to_db=True
            )
            
            return {
                'applicant_id': applicant_id,
                'success': True,
                'matches_found': result['total_matches'],
                'message': result['message'],
                'insufficient_data': result['insufficient_data']
            }
        except Exception as e:
            logger.error(f"Error processing applicant {applicant_id}: {e}")
            return {
                'applicant_id': applicant_id,
                'success': False,
                'error': str(e),
                'matches_found': 0,
                'insufficient_data': False
            }

    def process_single_job_matching(self, job_id: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Process cosine similarity matching for a single job"""
        try:
            logger.info(f"Processing cosine similarity matching for job: {job_id}")
            result = self.perform_comprehensive_cosine_matching_job_to_applicants(
                job_id, 
                threshold, 
                save_to_db=True
            )
            
            return {
                'job_id': job_id,
                'success': True,
                'matches_found': result['total_matches'],
                'message': result['message'],
                'total_applicants_processed': result.get('total_applicants_processed', 0),
                'valid_applicants': result.get('valid_applicants', 0)
            }
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            return {
                'job_id': job_id,
                'success': False,
                'error': str(e),
                'matches_found': 0
            }

    def batch_process_all_applicants(self, threshold: float = 0.5, max_profiles: int = None) -> Dict[str, Any]:
        """Process cosine similarity matching for all applicant profiles in batches"""
        start_time = time.time()
        
        try:
            profiles = self.get_all_applicant_profiles(limit=max_profiles or 1000)
            
            if not profiles:
                return {
                    'success': False,
                    'message': 'No applicant profiles found',
                    'total_processed': 0,
                    'total_matches': 0
                }
            
            logger.info(f"Starting batch cosine similarity matching for {len(profiles)} applicants")
            
            results = []
            total_matches = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_applicant = {
                    executor.submit(self.process_single_applicant_matching, profile['id'], threshold): profile['id']
                    for profile in profiles
                }
                
                for future in as_completed(future_to_applicant):
                    applicant_id = future_to_applicant[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result['success']:
                            total_matches += result['matches_found']
                    except Exception as e:
                        logger.error(f"Error processing applicant {applicant_id}: {e}")
                        results.append({
                            'applicant_id': applicant_id,
                            'success': False,
                            'error': str(e),
                            'matches_found': 0
                        })
            
            successful_processes = [r for r in results if r['success']]
            failed_processes = [r for r in results if r['success'] == False]
            insufficient_data = [r for r in results if r.get('insufficient_data', False)]
            
            processing_time = time.time() - start_time
            
            summary = {
                'success': True,
                'total_applicants': len(profiles),
                'total_processed': len(results),
                'successful_processes': len(successful_processes),
                'failed_processes': len(failed_processes),
                'insufficient_data_profiles': len(insufficient_data),
                'total_matches_generated': total_matches,
                'processing_time_seconds': round(processing_time, 2),
                'average_time_per_applicant': round(processing_time / len(profiles), 2) if profiles else 0,
                'threshold_used': threshold
            }
            
            logger.info(f"Batch processing completed: {summary}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in batch processing all applicants: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_processed': 0,
                'total_matches': 0
            }

    def batch_process_all_jobs(self, threshold: float = 0.5, max_jobs: int = None) -> Dict[str, Any]:
        """Process cosine similarity matching for all job postings in batches"""
        start_time = time.time()
        
        try:
            jobs = self.get_all_job_postings(limit=max_jobs or 100)
            
            if not jobs:
                return {
                    'success': False,
                    'message': 'No job postings found',
                    'total_processed': 0,
                    'total_matches': 0
                }
            
            logger.info(f"Starting batch cosine similarity matching for {len(jobs)} jobs")
            
            results = []
            total_matches = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_job = {
                    executor.submit(self.process_single_job_matching, job['id'], threshold): job['id']
                    for job in jobs
                }
                
                for future in as_completed(future_to_job):
                    job_id = future_to_job[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result['success']:
                            total_matches += result['matches_found']
                    except Exception as e:
                        logger.error(f"Error processing job {job_id}: {e}")
                        results.append({
                            'job_id': job_id,
                            'success': False,
                            'error': str(e),
                            'matches_found': 0
                        })
            
            successful_processes = [r for r in results if r['success']]
            failed_processes = [r for r in results if r['success'] == False]
            
            processing_time = time.time() - start_time
            
            summary = {
                'success': True,
                'total_jobs': len(jobs),
                'total_processed': len(results),
                'successful_processes': len(successful_processes),
                'failed_processes': len(failed_processes),
                'total_matches_generated': total_matches,
                'processing_time_seconds': round(processing_time, 2),
                'average_time_per_job': round(processing_time / len(jobs), 2) if jobs else 0,
                'threshold_used': threshold
            }
            
            logger.info(f"Batch processing completed: {summary}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in batch processing all jobs: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_processed': 0,
                'total_matches': 0
            }

class ResourceAwareMatcher(JobApplicantMatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_batch_size = kwargs.get('max_batch_size', 50)
        self.request_semaphore = threading.Semaphore(kwargs.get('max_concurrent_requests', 10))
    
    def batch_encode_texts_with_backoff(self, texts: List[str], batch_size: int = 64, max_retries: int = 3) -> np.ndarray:
        """Batch encoding with exponential backoff and resource management"""
        for attempt in range(max_retries):
            try:
                with self.request_semaphore:
                    return self.batch_encode_texts(texts, batch_size)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Batch encoding failed after {max_retries} attempts: {e}")
                    return np.array([])
                
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Batch encoding attempt {attempt + 1} failed, retrying in {wait_time:.2f}s: {e}")
                time.sleep(wait_time)

# ==================== APPLICATION STARTUP OPTIMIZATION ====================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize matcher in background to prevent cold start blocking
matcher = None
matcher_initialization_started = False

def initialize_matcher_background():
    """Initialize matcher in background thread"""
    global matcher, matcher_initialization_started
    
    if matcher_initialization_started:
        return
    
    matcher_initialization_started = True
    logger.info("Starting background matcher initialization...")
    
    try:
        # Initialize with lightweight model
        global_matcher = JobApplicantMatcher(
            SUPABASE_URL, 
            SUPABASE_KEY, 
            'BAAI/bge-small-en-v1.5'  # Use small model
        )
        
        # Try to warm up the engine
        try:
            engine_ready = global_matcher.ensure_engine_ready(timeout=90)
            if engine_ready:
                logger.info("✓ Matcher initialized and warmed up successfully")
            else:
                logger.warning("Matcher initialized but warmup incomplete")
        except Exception as warmup_error:
            logger.error(f"Matcher warmup failed: {warmup_error}")
        
        matcher = global_matcher
        
    except Exception as e:
        logger.error(f"Failed to initialize matcher in background: {e}")
        matcher = None

# Start background initialization immediately
threading.Thread(target=initialize_matcher_background, daemon=True).start()

try:
    matcher = JobApplicantMatcher(SUPABASE_URL, SUPABASE_KEY, 'BAAI/bge-large-en-v1.5')
    logger.info("Cosine similarity matcher initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize cosine similarity matcher: {e}")
    matcher = None

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'running',
        'service': 'Semantic Matching API',
        'version': '2.0.0',
        'matching_mode': 'Unified Cosine Similarity (Skill Filtering Optional)',
        'note': 'Use use_skill_filtering parameter to enable skill filtering',
        'endpoints': {
            'health': '/api/health',
            'cosine_matching': '/api/cosine-matching (POST) - use use_skill_filtering parameter',
            'batch_applicants': '/api/batch-cosine-matching/applicants (POST)',
            'batch_jobs': '/api/batch-cosine-matching/jobs (POST)',
            'job_matching': '/api/cosine-matching/job/<job_id> (POST)',
            'my_matches': '/api/my-cosine-matches (GET)',
            'stats': '/api/cosine-stats (GET)'
        },
        'parameters': {
            'cosine_matching': {
                'use_skill_filtering': 'boolean (default: false) - Enable to only consider matching skills',
                'threshold': 'float (default: 0.5) - Similarity threshold',
                'top_n': 'integer (default: 20) - Number of top matches to return',
                'save_to_db': 'boolean (default: true) - Save matches to database'
            }
        }
    })
@app.route('/api', methods=['GET'])
def api_root():
    return jsonify({
        'message': 'Semantic Matching API',
        'available_endpoints': [
            'POST /api/cosine-matching (use use_skill_filtering parameter)',
            'POST /api/batch-cosine-matching/applicants', 
            'POST /api/batch-cosine-matching/jobs',
            'POST /api/cosine-matching/job/<job_id>',
            'GET /api/my-cosine-matches',
            'GET /api/cosine-stats',
            'GET /api/health'
        ],
        'cosine_matching_parameters': {
            'use_skill_filtering': 'Set to true for skill-filtered matching, false for standard matching',
            'threshold': 'Similarity threshold (0.0-1.0)',
            'top_n': 'Number of top matches to return',
            'save_to_db': 'Whether to save matches to database'
        }
    })
@app.route('/api/cosine-matching', methods=['POST'])
def cosine_matching():
    """
    Perform comprehensive cosine similarity matching - single applicant to all jobs
    Now includes both standard and skill-filtered matching in one endpoint
    """
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        data = request.get_json() or {}
        
        if data.get('batch_process', False):
            return batch_cosine_matching_applicants()
        
        user_id = request.headers.get('User-Id') or data.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID not provided'}), 400
        
        threshold = float(data.get('threshold', 0.5))
        top_n = int(data.get('top_n', 20))
        save_to_db = data.get('save_to_db', True)
        
        use_skill_filtering = data.get('use_skill_filtering', False)  
        
        start_time = time.time()
        
        if use_skill_filtering:
            logger.info(f"Using SKILL-FILTERED matching for user {user_id}")
            matching_result = matcher.perform_filtered_matching_applicant_to_jobs(
                user_id, threshold, save_to_db
            )
            matching_engine = 'filtered_cosine_similarity'
            matching_method = matching_result.get('matching_method', 'filtered_skills_only')
        else:
            logger.info(f"Using STANDARD matching for user {user_id}")
            matching_result = matcher.perform_comprehensive_cosine_matching_applicant_to_jobs(
                user_id, threshold, save_to_db
            )
            matching_engine = 'comprehensive_cosine_similarity'
            matching_method = 'standard_all_skills'
        

        top_matches = matcher.get_top_cosine_matches(matching_result['matches'], top_n)
        response_time = time.time() - start_time
        
        serializable_response = convert_to_serializable({
            'success': True,
            'data': {
                'total_matches': matching_result['total_matches'],
                'top_matches': top_matches,
                'threshold_used': threshold,
                'response_time_seconds': round(response_time, 2),
                'saved_to_db': save_to_db,
                'matching_engine': matching_engine,
                'matching_method': matching_method,
                'skill_filtering_enabled': use_skill_filtering,
                'matching_direction': 'one_applicant_to_all_jobs',
                'table_used': 'job_match_notification',
                'insufficient_data': matching_result['insufficient_data'],
                'message': matching_result['message']
            }
        })
        
        return jsonify(serializable_response)
        
    except Exception as e:
        logger.error(f"Error in cosine similarity matching: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/batch-cosine-matching/applicants', methods=['POST'])
def batch_cosine_matching_applicants():
    """Perform cosine similarity matching for all applicant profiles"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        data = request.get_json() or {}
        threshold = float(data.get('threshold', 0.5))
        max_profiles = data.get('max_profiles')
        use_skill_filtering = data.get('use_skill_filtering', False)
        
        if use_skill_filtering:
            result = matcher.batch_process_all_applicants_optimized(threshold, max_profiles)
        else:
            result = matcher.batch_process_all_applicants(threshold, max_profiles)
        
        return jsonify(convert_to_serializable(result))
        
    except Exception as e:
        logger.error(f"Error in batch cosine similarity matching: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/batch-cosine-matching', methods=['POST'])
def batch_cosine_matching_legacy():
    """Legacy endpoint for backward compatibility - redirects to applicants batch"""
    return batch_cosine_matching_applicants()

@app.route('/api/cosine-matching/job/<job_id>', methods=['POST'])
def cosine_matching_for_job(job_id):
    """Perform comprehensive cosine similarity matching between one job and all applicants"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        data = request.get_json() or {}
        threshold = float(data.get('threshold', 0.5))
        top_n = int(data.get('top_n', 20))
        save_to_db = data.get('save_to_db', True)
        
        start_time = time.time()
        matching_result = matcher.perform_comprehensive_cosine_matching_job_to_applicants(job_id, threshold, save_to_db)
        top_matches = matcher.get_top_cosine_matches(matching_result['matches'], top_n)
        response_time = time.time() - start_time
        
        serializable_response = convert_to_serializable({
            'success': True,
            'data': {
                'job_id': job_id,
                'total_matches': matching_result['total_matches'],
                'top_matches': top_matches,
                'total_applicants_processed': matching_result.get('total_applicants_processed', 0),
                'valid_applicants': matching_result.get('valid_applicants', 0),
                'threshold_used': threshold,
                'response_time_seconds': round(response_time, 2),
                'saved_to_db': save_to_db,
                'matching_engine': 'comprehensive_cosine_similarity',
                'matching_direction': 'one_job_to_all_applicants',
                'table_used': 'applicant_match',
                'insufficient_data': matching_result['insufficient_data'],
                'message': matching_result['message'],
                'insufficient_data_applicants': matching_result.get('insufficient_data_applicants', [])
            }
        })
        
        return jsonify(serializable_response)
    except Exception as e:
        logger.error(f"Error in cosine similarity matching for job {job_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/batch-cosine-matching/jobs', methods=['POST'])
def batch_cosine_matching_jobs():
    """Perform cosine similarity matching for all job postings"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        data = request.get_json() or {}
        threshold = float(data.get('threshold', 0.5))
        max_jobs = data.get('max_jobs')
        
        result = matcher.batch_process_all_jobs(threshold, max_jobs)
        
        return jsonify(convert_to_serializable(result))
        
    except Exception as e:
        logger.error(f"Error in batch cosine similarity matching for jobs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/my-cosine-matches', methods=['GET'])
def get_my_cosine_matches():
    """Get cosine similarity matches for the current user (applicant to jobs)"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        user_id = request.headers.get('User-Id')
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID not provided'}), 400
        
        matches_response = matcher.supabase.table('job_match_notification') \
            .select('''
                job_id, 
                similarity_score,
                cosine_score,
                skill_score,
                experience_score,
                match_strength,
                jobpost (*)
            ''') \
            .eq('applicant_id', user_id) \
            .order('similarity_score', ascending=False) \
            .execute()
        
        return jsonify({
            'success': True,
            'matches': matches_response.data if matches_response.data else [],
            'matching_type': 'cosine_similarity',
            'matching_direction': 'one_applicant_to_all_jobs',
            'table_used': 'job_match_notification'
        })
    except Exception as e:
        logger.error(f"Error getting user cosine matches: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/job-matches/<job_id>', methods=['GET'])
def get_job_matches(job_id):
    """Get matches for a specific job (job to applicants)"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        matches_response = matcher.supabase.table('applicant_match') \
            .select('''
                applicant_id, 
                similarity_score,
                cosine_score,
                skill_score,
                experience_score,
                match_strength,
                applicant_profiles (*)
            ''') \
            .eq('job_id', job_id) \
            .order('similarity_score', ascending=False) \
            .execute()
        
        return jsonify({
            'success': True,
            'matches': matches_response.data if matches_response.data else [],
            'matching_type': 'cosine_similarity',
            'matching_direction': 'one_job_to_all_applicants',
            'table_used': 'applicant_match'
        })
    except Exception as e:
        logger.error(f"Error getting job matches for {job_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/applicant-stats', methods=['GET'])
def get_applicant_stats():
    """Get statistics about applicant profiles and their matching status"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        applicants_response = matcher.supabase.table('applicant_profiles') \
            .select('id, created_at', count='exact') \
            .execute()
        
        matches_response = matcher.supabase.table('job_match_notification') \
            .select('applicant_id', count='exact') \
            .execute()
        
        unique_applicants_with_matches = set()
        if matches_response.data:
            for match in matches_response.data:
                unique_applicants_with_matches.add(match['applicant_id'])
        
        total_applicants = applicants_response.count if hasattr(applicants_response, 'count') else len(applicants_response.data)
        applicants_with_matches = len(unique_applicants_with_matches)
        applicants_without_matches = total_applicants - applicants_with_matches
        
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        recent_applicants_response = matcher.supabase.table('applicant_profiles') \
            .select('id', count='exact') \
            .gte('created_at', week_ago) \
            .execute()
        
        recent_applicants = recent_applicants_response.count if hasattr(recent_applicants_response, 'count') else len(recent_applicants_response.data)
        
        return jsonify({
            'success': True,
            'data': {
                'total_applicants': total_applicants,
                'applicants_with_matches': applicants_with_matches,
                'applicants_without_matches': applicants_without_matches,
                'match_coverage_percentage': round((applicants_with_matches / total_applicants * 100) if total_applicants > 0 else 0, 2),
                'recent_applicants_last_7_days': recent_applicants,
                'match_table_used': 'job_match_notification'
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting applicant stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/job-stats', methods=['GET'])
def get_job_stats():
    """Get statistics about job postings and their matching status"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        jobs_response = matcher.supabase.table('jobpost') \
            .select('id, created_at', count='exact') \
            .execute()
        
        active_jobs_response = matcher.supabase.table('jobpost') \
            .select('id', count='exact') \
            .eq('status', 'Open') \
            .eq('approval_status', 'approved') \
            .execute()
        
        matches_response = matcher.supabase.table('applicant_match') \
            .select('job_id', count='exact') \
            .execute()
        
        unique_jobs_with_matches = set()
        if matches_response.data:
            for match in matches_response.data:
                unique_jobs_with_matches.add(match['job_id'])
        
        total_jobs = jobs_response.count if hasattr(jobs_response, 'count') else len(jobs_response.data)
        active_jobs = active_jobs_response.count if hasattr(active_jobs_response, 'count') else len(active_jobs_response.data)
        jobs_with_matches = len(unique_jobs_with_matches)
        jobs_without_matches = active_jobs - jobs_with_matches
        
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        recent_jobs_response = matcher.supabase.table('jobpost') \
            .select('id', count='exact') \
            .gte('created_at', week_ago) \
            .execute()
        
        recent_jobs = recent_jobs_response.count if hasattr(recent_jobs_response, 'count') else len(recent_jobs_response.data)
        
        return jsonify({
            'success': True,
            'data': {
                'total_jobs': total_jobs,
                'active_jobs': active_jobs,
                'jobs_with_matches': jobs_with_matches,
                'jobs_without_matches': jobs_without_matches,
                'match_coverage_percentage': round((jobs_with_matches / active_jobs * 100) if active_jobs > 0 else 0, 2),
                'recent_jobs_last_7_days': recent_jobs,
                'match_table_used': 'applicant_match'
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting job stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cosine-stats', methods=['GET'])
def get_cosine_stats():
    """Get cosine similarity matching statistics"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        applicants = matcher.supabase.table('applicant_profiles').select('id', count='exact').execute()
        jobs = matcher.supabase.table('jobpost').select('id', count='exact').execute()
        
        job_matches = matcher.supabase.table('job_match_notification').select('id', count='exact').execute()
        applicant_matches = matcher.supabase.table('applicant_match').select('id', count='exact').execute()
        
        total_matches = (job_matches.count if hasattr(job_matches, 'count') else len(job_matches.data)) + \
                       (applicant_matches.count if hasattr(applicant_matches, 'count') else len(applicant_matches.data))
        
        job_avg_scores = matcher.supabase.table('job_match_notification') \
            .select('similarity_score, cosine_score, skill_score, experience_score') \
            .execute()
        
        applicant_avg_scores = matcher.supabase.table('applicant_match') \
            .select('similarity_score, cosine_score, skill_score, experience_score') \
            .execute()
        
        all_scores = []
        if job_avg_scores.data:
            all_scores.extend(job_avg_scores.data)
        if applicant_avg_scores.data:
            all_scores.extend(applicant_avg_scores.data)
        
        if all_scores:
            avg_similarity = np.mean([m['similarity_score'] for m in all_scores])
            avg_cosine = np.mean([m['cosine_score'] for m in all_scores])
            avg_skill = np.mean([m['skill_score'] for m in all_scores])
            avg_experience = np.mean([m.get('experience_score', 0) for m in all_scores])
        else:
            avg_similarity = avg_cosine = avg_skill = avg_experience = 0
        
        return jsonify({
            'success': True,
            'data': convert_to_serializable({
                'applicants_count': applicants.count if hasattr(applicants, 'count') else 0,
                'jobs_count': jobs.count if hasattr(jobs, 'count') else 0,
                'matches_count': total_matches,
                'job_match_notification_count': job_matches.count if hasattr(job_matches, 'count') else len(job_matches.data),
                'applicant_match_count': applicant_matches.count if hasattr(applicant_matches, 'count') else len(applicant_matches.data),
                'average_scores': {
                    'similarity': avg_similarity,
                    'cosine': avg_cosine,
                    'skill': avg_skill,
                    'experience': avg_experience
                }
            })
        })
    except Exception as e:
        logger.error(f"Error getting cosine stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/force-rematch-all', methods=['POST'])
def force_rematch_all():
    """Force re-match all applicants and jobs (clear existing matches and recalculate)"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        delete_response1 = matcher.supabase.table('job_match_notification') \
            .delete() \
            .neq('id', '00000000-0000-0000-0000-000000000000') \
            .execute()
        
        delete_response2 = matcher.supabase.table('applicant_match') \
            .delete() \
            .neq('id', '00000000-0000-0000-0000-000000000000') \
            .execute()
        
        logger.info("Cleared all existing matches from both tables")
        
        data = request.get_json() or {}
        threshold = float(data.get('threshold', 0.5))
        
        applicant_result = matcher.batch_process_all_applicants(threshold)
        
        job_result = matcher.batch_process_all_jobs(threshold)
        
        return jsonify(convert_to_serializable({
            'success': True,
            'cleared_existing_matches': True,
            'tables_cleared': ['job_match_notification', 'applicant_match'],
            'applicant_rematch_results': applicant_result,
            'job_rematch_results': job_result
        }))
        
    except Exception as e:
        logger.error(f"Error in force rematch: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cosine-matching/status/<match_id>', methods=['GET'])
def get_match_status(match_id):
    """Get status of a specific matching process"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:

        return jsonify({
            'success': True,
            'match_id': match_id,
            'status': 'completed', 
            'message': 'Match status endpoint - implement job tracking system'
        })
    except Exception as e:
        logger.error(f"Error getting match status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cosine-matching/history', methods=['GET'])
def get_matching_history():
    """Get history of matching operations"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
       
        return jsonify({
            'success': True,
            'history': [],
            'message': 'Matching history endpoint - implement audit trail system'
        })
    except Exception as e:
        logger.error(f"Error getting matching history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear caches (admin endpoint)"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        matcher.job_cache.clear()
        matcher.profile_cache.clear()
        matcher.embedding_cache.clear()
        return jsonify({'success': True, 'message': 'Cosine similarity caches cleared'})
    except Exception as e:
        logger.error(f"Error clearing cosine similarity cache: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/semantic-matching', methods=['POST'])
def semantic_matching():
    """Backward compatibility endpoint - redirects to cosine matching"""
    return cosine_matching()

@app.route('/api/my-semantic-matches', methods=['GET'])
def get_my_semantic_matches():
    """Backward compatibility endpoint - redirects to cosine matches"""
    return get_my_cosine_matches()

@app.route('/api/semantic-stats', methods=['GET'])
def get_semantic_stats():
    """Backward compatibility endpoint - redirects to cosine stats"""
    return get_cosine_stats()

@app.route('/api/batch-semantic-matching', methods=['POST'])
def batch_semantic_matching():
    """Backward compatibility endpoint - redirects to batch cosine matching"""
    return batch_cosine_matching_applicants()

@app.route('/api/my-matches', methods=['GET'])
def get_my_matches():
    """Get matches for the current user (backward compatibility)"""
    return get_my_cosine_matches()

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get matching statistics (backward compatibility)"""
    return get_cosine_stats()
@app.route('/api/health', methods=['GET'])
def health_check():
    """Detailed health check with model status"""
    health_data = {
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system': {
            'python_version': sys.version,
            'platform': platform.platform()
        },
        'matcher': {
            'initialized': matcher is not None,
            'model_status': 'unknown'
        },
        'cache': {
            'persistent_enabled': True,
            'cache_dir': persistent_cache.cache_dir
        }
    }
    
    if matcher:
        # Check model status
        model_ready = False
        if hasattr(matcher, 'model_initialized'):
            model_ready = matcher.model_initialized
            if matcher.semantic_engine:
                model_ready = model_ready and matcher.semantic_engine.model_loaded
        
        health_data['matcher'].update({
            'model_status': 'ready' if model_ready else 'loading',
            'model_initialized': matcher.model_initialized if hasattr(matcher, 'model_initialized') else False,
            'engine_ready': bool(matcher.semantic_engine),
            'model_loaded': matcher.semantic_engine.model_loaded if matcher.semantic_engine else False,
            'model_name': matcher.model_name if hasattr(matcher, 'model_name') else 'unknown'
        })
    
    return jsonify(health_data)

@app.route('/api/warmup', methods=['POST'])
def warmup():
    """Force warmup of the system"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Matcher not initialized yet'}), 503
    
    try:
        engine_ready = matcher.ensure_engine_ready(timeout=120)
        
        if engine_ready and matcher.semantic_engine:
            test_embedding = matcher.semantic_engine.get_semantic_embedding("test warmup")
            test_success = test_embedding is not None and test_embedding.size > 0
        else:
            test_success = False
        
        return jsonify({
            'success': True,
            'engine_ready': engine_ready,
            'test_embedding_generated': test_success,
            'message': 'Warmup completed' if engine_ready else 'Warmup in progress'
        })
        
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Warmup failed'
        }), 500
if __name__ == '__main__':
    if matcher:
        logger.info("Attempting final warmup before serving...")
        try:
            matcher.ensure_engine_ready(timeout=30)
        except Exception as e:
            logger.warning(f"Final warmup incomplete: {e}")
    
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=False,
        threaded=True,
        processes=1 
    )
