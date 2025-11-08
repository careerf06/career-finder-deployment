from concurrent.futures import ThreadPoolExecutor, as_completed
import random
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

class SemanticSimilarityEngine:
    """Enhanced semantic similarity engine with manual cosine similarity"""
    
    def __init__(self, model_name: str = 'thenlper/gte-large'):
        try:
            cache_dir = os.getenv('TRANSFORMERS_CACHE', '/tmp/transformers_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            self.model = SentenceTransformer(
                model_name,
                device='cpu',
                use_auth_token=False,
                cache_folder=cache_dir
            )
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            logger.info(f"SemanticSimilarityEngine initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SemanticSimilarityEngine: {e}")
            raise
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Using basic tokenization.")
            self.nlp = None
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def get_semantic_embedding(self, text: str) -> np.ndarray:
        """Get semantic embedding for text using cosine similarity compatible format"""
        if not text or not text.strip():
            return np.zeros(1024, dtype=np.float32) 
        try:
            with torch.no_grad():
                embedding = self.model.encode(
                    text, 
                    convert_to_tensor=False, 
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(1024, dtype=np.float32)
    
    def manual_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Manual cosine similarity calculation using the formula: (A·B) / (||A|| * ||B||)"""
        try:
            if embedding1.size == 0 or embedding2.size == 0:
                return 0.0
            
            if embedding1.ndim > 1:
                embedding1 = embedding1.flatten()
            if embedding2.ndim > 1:
                embedding2 = embedding2.flatten()
            
            # Ensure both embeddings have the same dimensions
            min_dim = min(embedding1.shape[0], embedding2.shape[0])
            embedding1 = embedding1[:min_dim]
            embedding2 = embedding2[:min_dim]
            
            # Step 1: Calculate dot product (A · B)
            dot_product = np.dot(embedding1, embedding2)
            
            # Step 2: Calculate magnitudes (||A|| and ||B||)
            magnitude_a = np.linalg.norm(embedding1)
            magnitude_b = np.linalg.norm(embedding2)
            
            # Step 3: Avoid division by zero
            if magnitude_a == 0 or magnitude_b == 0:
                return 0.0
            
            # Step 4: Calculate cosine similarity: (A·B) / (||A|| * ||B||)
            cosine_similarity = dot_product / (magnitude_a * magnitude_b)
            
            # Ensure the result is between -1 and 1 (due to floating point precision)
            cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
            
            logger.debug(f"Cosine similarity calculation: dot_product={dot_product:.4f}, "
                        f"magnitude_a={magnitude_a:.4f}, magnitude_b={magnitude_b:.4f}, "
                        f"result={cosine_similarity:.4f}")
            
            return float(cosine_similarity)
            
        except Exception as e:
            logger.error(f"Error in manual cosine similarity calculation: {e}")
            return 0.0
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings using manual computation"""
        return self.manual_cosine_similarity(embedding1, embedding2)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using cosine similarity on embeddings"""
        if not text1.strip() or not text2.strip():
            return 0.0
            
        try:
            embedding1 = self.get_semantic_embedding(text1)
            embedding2 = self.get_semantic_embedding(text2)
            
            similarity = self.calculate_cosine_similarity(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}")
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
                
            similarity = intersection / union
            return float(similarity)
        except Exception as e:
            logger.error(f"Error in keyword similarity calculation: {e}")
            return 0.0
    
    def calculate_hybrid_similarity(self, text1: str, text2: str, 
                                  semantic_weight: float = 0.7, 
                                  keyword_weight: float = 0.3) -> float:
        """Calculate hybrid similarity combining cosine semantic and keyword approaches"""
        try:
            semantic_sim = self.calculate_semantic_similarity(text1, text2)
            keyword_sim = self.calculate_keyword_similarity(text1, text2)
            
            hybrid_score = (semantic_weight * semantic_sim) + (keyword_weight * keyword_sim)
            return float(hybrid_score)
        except Exception as e:
            logger.error(f"Error in hybrid similarity calculation: {e}")
            return 0.0
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for keyword analysis"""
        if not text:
            return []
        
        try:
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            
            if self.nlp:
                doc = self.nlp(text)
                tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
            else:
                tokens = word_tokenize(text)
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                         if token.lower() not in self.stop_words and token.isalpha()]
            
            return tokens
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            return []

class STSConfig:
    """Configuration for Semantic Textual Similarity system"""
    
    def __init__(self):
        self.model_name = os.getenv('STS_MODEL_NAME', 'thenlper/gte-large')
        self.similarity_threshold = float(os.getenv('STS_SIMILARITY_THRESHOLD', '0.5'))
        self.batch_size = int(os.getenv('STS_BATCH_SIZE', '64'))
        self.max_workers = int(os.getenv('STS_MAX_WORKERS', '4'))
        self.cache_ttl_minutes = int(os.getenv('STS_CACHE_TTL', '1'))
        self.embedding_dimension = 1024 
        
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
    def __init__(self, supabase_url: str, supabase_key: str, model_name: str = 'thenlper/gte-large'):
        """Initialize with Supabase connection and cosine similarity engine"""
        try:
            if not supabase_url or not supabase_key:
                raise ValueError("Supabase URL and Key must be provided")
                
            self.supabase: Client = create_client(supabase_url, supabase_key)
            
            self.semantic_engine = SemanticSimilarityEngine(model_name)
            
            self.job_cache = {}
            self.profile_cache = {}
            self.embedding_cache = {}
            self.cache_ttl = timedelta(minutes=1)
            
            self.batch_size = 10 
            self.max_workers = 4 
            
            self.metrics = STSMetrics()
            
            logger.info(f"Initialized JobApplicantMatcher with cosine similarity engine: {model_name}")
            
            self.test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize JobApplicantMatcher: {e}")
            raise

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

    def create_semantic_text_representation(self, profile: Dict, entity_type: str = "applicant") -> str:
        """Create optimized semantic text representation with enhanced context"""
        try:
            if entity_type == "applicant":
                skills = self._normalize_skills(profile.get('skills', []) or [])
                parts = [
                    f"Position: {profile.get('position', '')}",
                    f"Company: {profile.get('company', '')}",
                    f"Description: {self._clean_text(profile.get('description', ''))}",
                    f"Skills: {', '.join(skills[:15])}",
                    f"Industry: {profile.get('industry', '')}",
                    f"Education: {profile.get('education', '')}"
                ]
            else:
                requirements = self._normalize_skills(profile.get('requirements', []) or [])
                parts = [
                    f"Title: {profile.get('title', '')}",
                    f"Company: {profile.get('company_name', '')}",
                    f"Description: {self._clean_text(profile.get('description', ''))}",
                    f"Requirements: {', '.join(requirements[:12])}",
                    f"Job Type: {profile.get('job_type', '')}",
                    f"Industry: {profile.get('industry', '')}"
                ]
            
            valid_parts = []
            for part in parts:
                try:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        if value.strip():
                            valid_parts.append(part)
                    else:
                        if part.strip():
                            valid_parts.append(part)
                except Exception as e:
                    logger.warning(f"Error processing part '{part}': {e}")
                    continue
            
            return " | ".join(valid_parts) if valid_parts else "No information available"
        except Exception as e:
            logger.error(f"Error creating semantic text representation: {e}")
            return "Error in text representation"

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

    def batch_encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Optimized batch encoding with cosine similarity compatible embeddings"""
        if not texts:
            return np.array([])
            
        cache_key = hashlib.md5("|".join(texts).encode()).hexdigest()
        now = datetime.now()
        
        if cache_key in self.embedding_cache:
            data, timestamp = self.embedding_cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return data
        
        try:
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                valid_batch_texts = [text for text in batch_texts if text and text.strip()]
                
                if not valid_batch_texts:
                    continue
                    
                with torch.no_grad():
                    batch_embeddings = self.semantic_engine.model.encode(
                        valid_batch_texts, 
                        convert_to_tensor=False,
                        normalize_embeddings=True,  
                        show_progress_bar=False,
                        batch_size=min(batch_size, len(valid_batch_texts))
                    )
                embeddings.append(batch_embeddings)
            
            if embeddings:
                result = np.vstack(embeddings)
                self.embedding_cache[cache_key] = (result, now)
                return result
            else:
                return np.array([])
        except Exception as e:
            logger.error(f"Error in batch encoding: {e}")
            return np.array([])

    def calculate_comprehensive_cosine_similarity_applicant_to_jobs(self, jobs: List[Dict], profile: Dict) -> List[float]:
        """Calculate comprehensive cosine similarity between jobs and profile"""
        if not jobs or not profile:
            return []
            
        try:
            job_texts = [self.create_semantic_text_representation(job, "job") for job in jobs]
            profile_text = self.create_semantic_text_representation(profile, "applicant")
            
            logger.info(f"Generating semantic embeddings for {len(job_texts)} jobs and profile...")
            
            job_embeddings = self.batch_encode_texts(job_texts, batch_size=64)
            
            if job_embeddings.size == 0:
                logger.warning("No job embeddings generated")
                return [0.0] * len(jobs)
            
            profile_embedding = self.semantic_engine.get_semantic_embedding(profile_text)
            
            if profile_embedding.size == 0:
                logger.warning("No profile embedding generated")
                return [0.0] * len(jobs)
            
            logger.info("Calculating comprehensive cosine similarity...")
            similarity_scores = []
            
            for i in range(len(job_embeddings)):
                if i < len(job_embeddings):
                    job_embedding = job_embeddings[i]
                    similarity = self.semantic_engine.calculate_cosine_similarity(
                        job_embedding, profile_embedding
                    )
                    similarity_scores.append(similarity)
                else:
                    similarity_scores.append(0.0)
            
            if len(similarity_scores) != len(jobs):
                logger.warning(f"Score count mismatch: {len(similarity_scores)} scores for {len(jobs)} jobs")
                if len(similarity_scores) < len(jobs):
                    similarity_scores.extend([0.0] * (len(jobs) - len(similarity_scores)))
                else:
                    similarity_scores = similarity_scores[:len(jobs)]
            
            return similarity_scores
        except Exception as e:
            logger.error(f"Error in comprehensive cosine similarity calculation: {e}")
            return [0.0] * len(jobs) if jobs else []

    def calculate_comprehensive_cosine_similarity_job_to_applicants(self, applicants: List[Dict], job: Dict) -> List[float]:
        """Calculate comprehensive cosine similarity between applicants and job"""
        if not applicants or not job:
            return []
            
        try:
            applicant_texts = [self.create_semantic_text_representation(applicant, "applicant") for applicant in applicants]
            job_text = self.create_semantic_text_representation(job, "job")
            
            logger.info(f"Generating semantic embeddings for {len(applicant_texts)} applicants and job...")
            
            applicant_embeddings = self.batch_encode_texts(applicant_texts, batch_size=64)
            
            if applicant_embeddings.size == 0:
                logger.warning("No applicant embeddings generated")
                return [0.0] * len(applicants)
            
            job_embedding = self.semantic_engine.get_semantic_embedding(job_text)
            
            if job_embedding.size == 0:
                logger.warning("No job embedding generated")
                return [0.0] * len(applicants)
            
            logger.info("Calculating comprehensive cosine similarity...")
            similarity_scores = []
            
            for i in range(len(applicant_embeddings)):
                if i < len(applicant_embeddings):
                    applicant_embedding = applicant_embeddings[i]
                    similarity = self.semantic_engine.calculate_cosine_similarity(
                        applicant_embedding, job_embedding
                    )
                    similarity_scores.append(similarity)
                else:
                    similarity_scores.append(0.0)
            
            if len(similarity_scores) != len(applicants):
                logger.warning(f"Score count mismatch: {len(similarity_scores)} scores for {len(applicants)} applicants")
                if len(similarity_scores) < len(applicants):
                    similarity_scores.extend([0.0] * (len(applicants) - len(similarity_scores)))
                else:
                    similarity_scores = similarity_scores[:len(applicants)]
            
            return similarity_scores
        except Exception as e:
            logger.error(f"Error in comprehensive cosine similarity calculation: {e}")
            return [0.0] * len(applicants) if applicants else []

    def calculate_semantic_skill_similarity(self, job_requirements: List[str], applicant_skills: List[str]) -> float:
        """Calculate semantic skill similarity using cosine similarity"""
        if not job_requirements or not applicant_skills:
            return 0.0
        
        try:
            job_reqs = set(self._normalize_skills(job_requirements))
            applicant_skills_set = set(self._normalize_skills(applicant_skills))
            
            if not job_reqs:
                return 0.0
            
            intersection = len(job_reqs.intersection(applicant_skills_set))
            union = len(job_reqs.union(applicant_skills_set))
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            job_skills_text = " ".join(list(job_reqs)[:20]) 
            applicant_skills_text = " ".join(list(applicant_skills_set)[:20])
            
            cosine_similarity_score = self.semantic_engine.calculate_semantic_similarity(
                job_skills_text, applicant_skills_text
            )
            
            coverage = intersection / len(job_reqs) if job_reqs else 0
            
            combined_score = (0.4 * coverage) + (0.4 * cosine_similarity_score) + (0.2 * jaccard_similarity)
            
            return float(min(combined_score, 1.0))
        except Exception as e:
            logger.error(f"Error calculating semantic skill similarity: {e}")
            return 0.0

    def calculate_experience_similarity(self, job: Dict, profile: Dict) -> float:
        """Calculate experience similarity using cosine similarity"""
        try:
            job_experience = self._extract_experience_from_job(job)
            applicant_experience = self._extract_experience_from_profile(profile)
            
            if not job_experience or not applicant_experience:
                return 0.5 
            
            experience_score = self._compare_experience_levels(job_experience, applicant_experience)
            
            exp_cosine_score = self.semantic_engine.calculate_semantic_similarity(
                job_experience.get('description', ''),
                applicant_experience.get('description', '')
            )
            
            combined_experience_score = (0.6 * experience_score) + (0.4 * exp_cosine_score)
            return float(min(combined_experience_score, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating experience similarity: {e}")
            return 0.3 

    def _extract_experience_from_job(self, job: Dict) -> Dict:
        """Extract experience requirements from job posting"""
        experience_info = {
            'level': job.get('experience_level', ''),
            'years': job.get('years_experience', 0),
            'description': job.get('description', '') + " " + " ".join(job.get('requirements', [])),
            'type': job.get('job_type', '')
        }
        return experience_info

    def _extract_experience_from_profile(self, profile: Dict) -> Dict:
        """Extract experience information from applicant profile"""
        experience_info = {
            'level': profile.get('experience_level', ''),
            'years': profile.get('years_of_experience', 0),
            'description': profile.get('description', '') + " " + profile.get('position', ''),
            'companies': profile.get('company', ''),
            'industry': profile.get('industry', '')
        }
        return experience_info

    def _compare_experience_levels(self, job_exp: Dict, applicant_exp: Dict) -> float:
        """Compare experience levels and years"""
        try:
            level_mapping = {
                'entry': 1, 'junior': 1,
                'mid': 2, 'intermediate': 2, 'medior': 2,
                'senior': 3, 'expert': 4, 'lead': 4, 'principal': 4
            }
            
            job_level = job_exp.get('level', '').lower()
            applicant_level = applicant_exp.get('level', '').lower()
            
            job_level_score = level_mapping.get(job_level, 2) 
            applicant_level_score = level_mapping.get(applicant_level, 2)
            
            level_diff = abs(job_level_score - applicant_level_score)
            level_score = max(0, 1 - (level_diff * 0.25))  
            job_years = job_exp.get('years', 0) or 0
            applicant_years = applicant_exp.get('years', 0) or 0
            
            if job_years == 0:
                years_score = 0.7 
            else:
                years_ratio = min(applicant_years / job_years, 2.0)
                years_score = min(years_ratio, 1.0) 
            
            experience_match = (0.6 * level_score) + (0.4 * years_score)
            return float(min(experience_match, 1.0))
            
        except Exception as e:
            logger.error(f"Error comparing experience levels: {e}")
            return 0.5

    def calculate_cosine_weighted_score(self, cosine_score: float, skill_score: float, experience_score: float = 0.5) -> Dict[str, float]:
        """Calculate comprehensive weighted score with cosine similarity emphasis"""
        try:
            primary_score = (0.75 * cosine_score) + (0.20 * skill_score) + (0.05 * experience_score)
            
            return {
                'similarity_score': float(max(0.0, min(1.0, primary_score))),
                'cosine_score': float(max(0.0, min(1.0, cosine_score))),
                'skill_score': float(max(0.0, min(1.0, skill_score))),
                'experience_score': float(max(0.0, min(1.0, experience_score)))
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
                    similarity_score = max(0.0, min(1.0, match['scores']['similarity_score']))
                    cosine_score = max(0.0, min(1.0, match['scores']['cosine_score']))
                    skill_score = max(0.0, min(1.0, match['scores']['skill_score']))
                    experience_score = max(0.0, min(1.0, match['scores'].get('experience_score', 0.0)))
                    
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
            
    def perform_comprehensive_cosine_matching_applicant_to_jobs(self, user_id: str, threshold: float = 0.5, save_to_db: bool = True) -> Dict[str, Any]:
        """Perform comprehensive cosine similarity matching - one applicant to all jobs"""
        start_time = time.time()
        
        try:
            jobs = self.get_job_postings()
            profile = self.get_applicant_profile(user_id)
            
            if not jobs:
                logger.warning("No jobs found for cosine similarity matching")
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
            
            logger.info(f"Starting cosine similarity matching for user {user_id} with {len(jobs)} jobs")
            
            cosine_scores = self.calculate_comprehensive_cosine_similarity_applicant_to_jobs(jobs, profile)
            
            if len(cosine_scores) == 0:
                logger.warning("No cosine similarity scores calculated")
                return {
                    'matches': [],
                    'insufficient_data': False,
                    'message': 'No cosine similarity matches could be calculated',
                    'total_matches': 0
                }
            
            matches = []
            for job_idx, job in enumerate(jobs):
                if job_idx >= len(cosine_scores):
                    logger.warning(f"Job index {job_idx} out of range for cosine scores")
                    continue
                    
                cosine_score = cosine_scores[job_idx]
                
                if cosine_score < threshold * 0.5:
                    continue
                    
                skill_score = self.calculate_semantic_skill_similarity(
                    job.get('requirements', []),
                    profile.get('skills', [])
                )

                experience_score = self.calculate_experience_similarity(job, profile)

                scores = self.calculate_cosine_weighted_score(cosine_score, skill_score, experience_score)
                                
                if scores['similarity_score'] >= threshold:
                    match_strength = self.get_cosine_match_strength(scores['similarity_score'])
                    
                    match_data = {
                        'job_id': job['id'],
                        'applicant_id': user_id,
                        'scores': scores,
                        'job_title': job.get('title', 'Unknown Title'),
                        'job_company': job.get('company_name', 'Unknown Company'),
                        'match_strength': match_strength,
                        'analysis': {
                            'cosine_interpretation': self.interpret_cosine_match(scores),
                            'key_strengths': self.identify_key_strengths(scores, job, profile),
                            'improvement_areas': self.identify_improvement_areas(scores, job, profile)
                        }
                    }
                    matches.append(match_data)
                    
                    # Save each match immediately to Supabase
                    if save_to_db:
                        try:
                            self.save_single_match_to_db(match_data, user_id=user_id)
                            logger.info(f"Immediately saved match for job {job['id']} to applicant {user_id} with score {scores['similarity_score']:.4f}")
                        except Exception as save_error:
                            logger.error(f"Failed to save immediate match for job {job['id']}: {save_error}")
            
            # Sort matches by similarity score
            sorted_matches = sorted(matches, key=lambda x: x['scores']['similarity_score'], reverse=True)
            
            processing_time = time.time() - start_time
            logger.info(f"Comprehensive cosine similarity matching completed in {processing_time:.2f}s. Found {len(sorted_matches)} matches.")
            
            return {
                'matches': sorted_matches,
                'insufficient_data': False,
                'message': f'Found {len(sorted_matches)} matches',
                'total_matches': len(sorted_matches),
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive cosine similarity matching: {e}")
            return {
                'matches': [],
                'insufficient_data': False,
                'message': f'Error during matching: {str(e)}',
                'total_matches': 0
            }
    def perform_comprehensive_cosine_matching_job_to_applicants(self, job_id: str, threshold: float = 0.5, save_to_db: bool = True) -> Dict[str, Any]:
        """Perform comprehensive cosine similarity matching for one job to all applicants"""
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
            
            logger.info(f"Starting cosine similarity matching for job {job_id} with {len(valid_applicants)} valid applicants out of {len(applicants)} total")
            
            cosine_scores = self.calculate_comprehensive_cosine_similarity_job_to_applicants(valid_applicants, job)
            
            if len(cosine_scores) == 0:
                logger.warning("No cosine similarity scores calculated")
                return {
                    'matches': [],
                    'insufficient_data': False,
                    'message': 'No cosine similarity matches could be calculated',
                    'total_matches': 0
                }
            
            matches = []
            for applicant_idx, applicant in enumerate(valid_applicants):
                if applicant_idx >= len(cosine_scores):
                    logger.warning(f"Applicant index {applicant_idx} out of range for cosine scores")
                    continue
                    
                cosine_score = cosine_scores[applicant_idx]
                
                if cosine_score < threshold * 0.5:
                    continue
                    
                skill_score = self.calculate_semantic_skill_similarity(
                    job.get('requirements', []),
                    applicant.get('skills', [])
                )

                experience_score = self.calculate_experience_similarity(job, applicant)

                scores = self.calculate_cosine_weighted_score(cosine_score, skill_score, experience_score)
                                
                if scores['similarity_score'] >= threshold:
                    match_strength = self.get_cosine_match_strength(scores['similarity_score'])
                    
                    match_data = {
                        'applicant_id': applicant['id'],
                        'job_id': job_id,
                        'scores': scores,
                        'applicant_name': f"{applicant.get('first_name', '')} {applicant.get('last_name', '')}".strip(),
                        'applicant_position': applicant.get('position', ''),
                        'match_strength': match_strength,
                        'analysis': {
                            'cosine_interpretation': self.interpret_cosine_match(scores),
                            'key_strengths': self.identify_key_strengths(scores, job, applicant),
                            'improvement_areas': self.identify_improvement_areas(scores, job, applicant)
                        }
                    }
                    matches.append(match_data)
                    
                    # Save each match immediately to Supabase
                    if save_to_db:
                        try:
                            self.save_single_match_to_db(match_data, job_id=job_id)
                            logger.info(f"Immediately saved match for applicant {applicant['id']} to job {job_id} with score {scores['similarity_score']:.4f}")
                        except Exception as save_error:
                            logger.error(f"Failed to save immediate match for applicant {applicant['id']}: {save_error}")
            
            # Sort matches by similarity score (even though they're already being saved individually)
            sorted_matches = sorted(matches, key=lambda x: x['scores']['similarity_score'], reverse=True)
            
            processing_time = time.time() - start_time
            logger.info(f"Comprehensive cosine similarity matching completed in {processing_time:.2f}s. Found {len(sorted_matches)} matches for job {job_id}.")
            
            return {
                'matches': sorted_matches,
                'insufficient_data': False,
                'message': f'Found {len(sorted_matches)} matches out of {len(valid_applicants)} valid applicants',
                'total_matches': len(sorted_matches),
                'total_applicants_processed': len(applicants),
                'valid_applicants': len(valid_applicants),
                'insufficient_data_applicants': insufficient_data_applicants,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive cosine similarity matching for job {job_id}: {e}")
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
                similarity_score = max(0.0, min(1.0, match['scores']['similarity_score']))
                cosine_score = max(0.0, min(1.0, match['scores']['cosine_score']))
                skill_score = max(0.0, min(1.0, match['scores']['skill_score']))
                experience_score = max(0.0, min(1.0, match['scores'].get('experience_score', 0.0)))
                
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
                
                # Save individual match immediately
                try:
                    result = self.supabase.table(table_name).upsert(match_entry).execute()
                    
                    if hasattr(result, 'error') and result.error:
                        logger.error(f"Database error saving single match to {table_name}: {result.error}")
                        # Try individual fallback
                        try:
                            individual_result = self.supabase.table(table_name).upsert(match_entry).execute()
                            if not (hasattr(individual_result, 'error') and individual_result.error):
                                logger.info(f"Individual fallback saved match to {table_name}")
                            else:
                                logger.error(f"Individual fallback also failed for {table_name}")
                        except Exception as individual_error:
                            logger.error(f"Individual fallback failed: {individual_error}")
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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    matcher = JobApplicantMatcher(SUPABASE_URL, SUPABASE_KEY, 'thenlper/gte-large')
    logger.info("Cosine similarity matcher initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize cosine similarity matcher: {e}")
    matcher = None


@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'running',
        'service': 'Semantic Matching API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'cosine_matching': '/api/cosine-matching (POST)',
            'batch_applicants': '/api/batch-cosine-matching/applicants (POST)',
            'batch_jobs': '/api/batch-cosine-matching/jobs (POST)',
            'job_matching': '/api/cosine-matching/job/<job_id> (POST)',
            'my_matches': '/api/my-cosine-matches (GET)',
            'stats': '/api/cosine-stats (GET)'
        }
    })

@app.route('/api', methods=['GET'])
def api_root():
    return jsonify({
        'message': 'Semantic Matching API',
        'available_endpoints': [
            'POST /api/cosine-matching',
            'POST /api/batch-cosine-matching/applicants', 
            'POST /api/batch-cosine-matching/jobs',
            'POST /api/cosine-matching/job/<job_id>',
            'GET /api/my-cosine-matches',
            'GET /api/cosine-stats',
            'GET /api/health'
        ]
    })

@app.route('/api/cosine-matching', methods=['POST'])
def cosine_matching():
    """Perform comprehensive cosine similarity matching - single applicant to all jobs"""
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
        
        start_time = time.time()
        matching_result = matcher.perform_comprehensive_cosine_matching_applicant_to_jobs(user_id, threshold, save_to_db)
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
                'matching_engine': 'comprehensive_cosine_similarity',
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
        process_only_new = data.get('process_only_new', False)
        hours_back = int(data.get('hours_back', 24))
        
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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'matcher_initialized': matcher is not None,
        'matching_engine': 'comprehensive_cosine_similarity',
        'supported_directions': ['one_applicant_to_all_jobs', 'one_job_to_all_applicants'],
        'tables_used': {
            'one_applicant_to_all_jobs': 'job_match_notification',
            'one_job_to_all_applicants': 'applicant_match'
        }
    })

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
