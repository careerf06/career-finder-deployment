from concurrent.futures import ThreadPoolExecutor, as_completed
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
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Filter out the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

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
    """Enhanced semantic similarity engine with cosine similarity"""
    
    def __init__(self, model_name: str = 'thenlper/gte-large'):
        try:
            self.model = SentenceTransformer(
                model_name,
                device='cpu',
                use_auth_token=False
            )
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            logger.info(f"SemanticSimilarityEngine initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SemanticSimilarityEngine: {e}")
            raise
        
        # Initialize NLP components with fallbacks
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
            return np.zeros(1024, dtype=np.float32)  # Default embedding size for gte-large
            
        try:
            with torch.no_grad():
                embedding = self.model.encode(
                    text, 
                    convert_to_tensor=False, 
                    normalize_embeddings=True,  # Crucial for cosine similarity
                    show_progress_bar=False
                )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(1024, dtype=np.float32)
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings using sklearn"""
        try:
            if embedding1.size == 0 or embedding2.size == 0:
                return 0.0
                
            # Ensure both embeddings are 2D arrays
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)
            
            # Ensure same dimensions
            min_dim = min(embedding1.shape[1], embedding2.shape[1])
            embedding1 = embedding1[:, :min_dim]
            embedding2 = embedding2[:, :min_dim]
            
            # Use sklearn's cosine_similarity for accurate calculation
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
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
            
            # Calculate Jaccard similarity
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
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            
            if self.nlp:
                doc = self.nlp(text)
                tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
            else:
                # Fallback to NLTK processing
                tokens = word_tokenize(text)
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                         if token.lower() not in self.stop_words and token.isalpha()]
            
            return tokens
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            return []

class JobApplicantMatcher:
    def __init__(self, supabase_url: str, supabase_key: str, model_name: str = 'thenlper/gte-large'):
        """Initialize with Supabase connection and cosine similarity engine"""
        try:
            if not supabase_url or not supabase_key:
                raise ValueError("Supabase URL and Key must be provided")
                
            self.supabase: Client = create_client(supabase_url, supabase_key)
            
            # Initialize semantic similarity engine with cosine similarity
            self.semantic_engine = SemanticSimilarityEngine(model_name)
            
            self.job_cache = {}
            self.profile_cache = {}
            self.embedding_cache = {}
            self.cache_ttl = timedelta(minutes=30)
            
            # Batch processing settings
            self.batch_size = 10  # Number of profiles to process simultaneously
            self.max_workers = 4  # Maximum concurrent threads
            
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

    def get_applicant_profiles(self, limit: int = 1000) -> List[Dict]:
        """Fetch all applicant profiles with caching"""
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
            logger.error(f"Error fetching applicant profiles: {e}")
            return []

    def has_sufficient_profile_data(self, profile: Dict) -> Tuple[bool, str]:
        """Check if applicant profile has sufficient data for meaningful matching"""
        if not profile:
            return False, "No profile data found"
        
        # Check for description
        description = (profile.get('description') or '').strip()
        has_description = len(description) > 50  # At least 50 characters
        
        # Check for skills
        skills = profile.get('skills') or []
        if isinstance(skills, str):
            try:
                skills = json.loads(skills) if skills.startswith('[') else [s.strip() for s in skills.split(',')]
            except:
                skills = [skills] if skills else []
        
        has_skills = len(skills) > 0 and any(len(str(skill).strip()) > 0 for skill in skills)
        
        # Check for position/company as additional context
        position = (profile.get('position') or '').strip()
        company = (profile.get('company') or '').strip()
        has_context = len(position) > 0 or len(company) > 0
        
        # Determine if data is sufficient
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
            else:  # job
                requirements = self._normalize_skills(profile.get('requirements', []) or [])
                parts = [
                    f"Title: {profile.get('title', '')}",
                    f"Company: {profile.get('company_name', '')}",
                    f"Description: {self._clean_text(profile.get('description', ''))}",
                    f"Requirements: {', '.join(requirements[:12])}",
                    f"Job Type: {profile.get('job_type', '')}",
                    f"Industry: {profile.get('industry', '')}"
                ]
            
            # Filter out empty parts safely
            valid_parts = []
            for part in parts:
                try:
                    # Split by colon and check if there's content after the colon
                    if ':' in part:
                        key, value = part.split(':', 1)
                        if value.strip():
                            valid_parts.append(part)
                    else:
                        # If no colon, just check if the part has content
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
            # Remove extra whitespace, special characters, and limit length
            cleaned = re.sub(r'[^\w\s\.\,\!]', '', text.strip())
            cleaned = re.sub(r'\s+', ' ', cleaned)
            return cleaned[:500]  # Increased limit for better semantic understanding
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
                    skills = [skills]
            
            normalized = []
            seen = set()
            for skill in skills:
                if isinstance(skill, str):
                    clean_skill = skill.lower().strip()
                    # Remove common prefixes/suffixes and standardize
                    clean_skill = re.sub(r'^(expert|proficient|skilled|experienced|basic)\s+in\s+', '', clean_skill)
                    clean_skill = re.sub(r'\s+(skills?|development|programming|framework|language)$', '', clean_skill)
                    
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
                # Filter out empty texts
                valid_batch_texts = [text for text in batch_texts if text and text.strip()]
                
                if not valid_batch_texts:
                    continue
                    
                with torch.no_grad():
                    batch_embeddings = self.semantic_engine.model.encode(
                        valid_batch_texts, 
                        convert_to_tensor=False,  # Keep as numpy for cosine similarity
                        normalize_embeddings=True,  # Crucial for cosine similarity
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

    def calculate_comprehensive_cosine_similarity(self, applicants: List[Dict], job: Dict) -> List[float]:
        """Calculate comprehensive cosine similarity between applicants and job"""
        if not applicants or not job:
            return []
            
        try:
            applicant_texts = [self.create_semantic_text_representation(applicant, "applicant") for applicant in applicants]
            job_text = self.create_semantic_text_representation(job, "job")
            
            logger.info(f"Generating semantic embeddings for {len(applicant_texts)} applicants and job...")
            
            # Batch process applicant embeddings for efficiency
            applicant_embeddings = self.batch_encode_texts(applicant_texts, batch_size=64)
            
            if applicant_embeddings.size == 0:
                logger.warning("No applicant embeddings generated")
                return [0.0] * len(applicants)
            
            # Get job embedding
            job_embedding = self.semantic_engine.get_semantic_embedding(job_text)
            
            if job_embedding.size == 0:
                logger.warning("No job embedding generated")
                return [0.0] * len(applicants)
            
            # Calculate cosine similarity using our engine
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
            
            # Ensure we have the right number of scores
            if len(similarity_scores) != len(applicants):
                logger.warning(f"Score count mismatch: {len(similarity_scores)} scores for {len(applicants)} applicants")
                # Pad or truncate to match applicant count
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
            
            # Traditional Jaccard similarity
            intersection = len(job_reqs.intersection(applicant_skills_set))
            union = len(job_reqs.union(applicant_skills_set))
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # Cosine similarity for skill sets
            job_skills_text = " ".join(list(job_reqs)[:20])  # Limit to prevent overly long texts
            applicant_skills_text = " ".join(list(applicant_skills_set)[:20])
            
            cosine_similarity_score = self.semantic_engine.calculate_semantic_similarity(
                job_skills_text, applicant_skills_text
            )
            
            # Coverage score (how many required skills the applicant has)
            coverage = intersection / len(job_reqs) if job_reqs else 0
            
            # Combined score with emphasis on coverage and semantic understanding
            combined_score = (0.4 * coverage) + (0.4 * cosine_similarity_score) + (0.2 * jaccard_similarity)
            
            return float(min(combined_score, 1.0))
        except Exception as e:
            logger.error(f"Error calculating semantic skill similarity: {e}")
            return 0.0

    def calculate_experience_similarity(self, job: Dict, applicant: Dict) -> float:
        """Calculate experience similarity using cosine similarity"""
        try:
            # Extract experience information
            job_experience = self._extract_experience_from_job(job)
            applicant_experience = self._extract_experience_from_applicant(applicant)
            
            if not job_experience or not applicant_experience:
                return 0.5  # Neutral score if no experience data
            
            # Calculate experience level match
            experience_score = self._compare_experience_levels(job_experience, applicant_experience)
            
            # Calculate cosine similarity of experience descriptions
            exp_cosine_score = self.semantic_engine.calculate_semantic_similarity(
                job_experience.get('description', ''),
                applicant_experience.get('description', '')
            )
            
            # Combined experience score
            combined_experience_score = (0.6 * experience_score) + (0.4 * exp_cosine_score)
            return float(min(combined_experience_score, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating experience similarity: {e}")
            return 0.3  # Default low score

    def _extract_experience_from_job(self, job: Dict) -> Dict:
        """Extract experience requirements from job posting"""
        experience_info = {
            'level': job.get('experience_level', ''),
            'years': job.get('years_experience', 0),
            'description': job.get('description', '') + " " + " ".join(job.get('requirements', [])),
            'type': job.get('job_type', '')
        }
        return experience_info

    def _extract_experience_from_applicant(self, applicant: Dict) -> Dict:
        """Extract experience information from applicant profile"""
        experience_info = {
            'level': applicant.get('experience_level', ''),
            'years': applicant.get('years_of_experience', 0),
            'description': applicant.get('description', '') + " " + applicant.get('position', ''),
            'companies': applicant.get('company', ''),
            'industry': applicant.get('industry', '')
        }
        return experience_info

    def _compare_experience_levels(self, job_exp: Dict, applicant_exp: Dict) -> float:
        """Compare experience levels and years"""
        try:
            # Map experience levels to numerical values
            level_mapping = {
                'entry': 1, 'junior': 1,
                'mid': 2, 'intermediate': 2, 'medior': 2,
                'senior': 3, 'expert': 4, 'lead': 4, 'principal': 4
            }
            
            job_level = job_exp.get('level', '').lower()
            applicant_level = applicant_exp.get('level', '').lower()
            
            job_level_score = level_mapping.get(job_level, 2)  # Default to mid-level
            applicant_level_score = level_mapping.get(applicant_level, 2)
            
            # Level match score
            level_diff = abs(job_level_score - applicant_level_score)
            level_score = max(0, 1 - (level_diff * 0.25))  # Penalize level differences
            
            # Years of experience match
            job_years = job_exp.get('years', 0) or 0
            applicant_years = applicant_exp.get('years', 0) or 0
            
            if job_years == 0:
                years_score = 0.7  # Neutral if job doesn't specify years
            else:
                years_ratio = min(applicant_years / job_years, 2.0)  # Cap at 2x required
                years_score = min(years_ratio, 1.0)  # More experience is good, but capped
            
            # Combined experience match
            experience_match = (0.6 * level_score) + (0.4 * years_score)
            return float(min(experience_match, 1.0))
            
        except Exception as e:
            logger.error(f"Error comparing experience levels: {e}")
            return 0.5

    def calculate_cosine_weighted_score(self, cosine_score: float, skill_score: float, experience_score: float = 0.5) -> Dict[str, float]:
        """Calculate comprehensive weighted score with cosine similarity emphasis"""
        try:
            # Updated primary score with experience factored in
            primary_score = (0.75 * cosine_score) + (0.20 * skill_score) + (0.05 * experience_score)
            
            # Return scores for database storage
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

    def get_existing_matches_for_job(self, job_id: str) -> Dict[str, Dict]:
        """Get existing matches for a job to prevent duplicates"""
        try:
            response = self.supabase.table('job_match_notification') \
                .select('applicant_id, similarity_score, match_strength, updated_at') \
                .eq('job_id', job_id) \
                .execute()
            
            existing_matches = {}
            for match in response.data:
                existing_matches[match['applicant_id']] = match
            return existing_matches
        except Exception as e:
            logger.error(f"Error fetching existing matches for job {job_id}: {e}")
            return {}

    def save_cosine_matches_to_db(self, matches: List[Dict], job_id: str):
        """Save cosine similarity matches with comprehensive scoring"""
        if not matches:
            logger.info("No matches to save to database")
            return
            
        try:
            existing_matches = self.get_existing_matches_for_job(job_id)
            match_data = []
            current_time = datetime.now().isoformat()
            
            for match in matches:
                applicant_id = match['applicant_id']
                existing_match = existing_matches.get(applicant_id)
                
                # Only include if it's a new match or score has significantly changed
                if (not existing_match or 
                    abs(existing_match['similarity_score'] - match['scores']['similarity_score']) > 0.01):
                    
                    match_data.append({
                        'applicant_id': applicant_id,
                        'job_id': job_id,
                        'similarity_score': match['scores']['similarity_score'],
                        'cosine_score': match['scores']['cosine_score'],
                        'skill_score': match['scores']['skill_score'],
                        'experience_score': match['scores'].get('experience_score', 0.0),
                        'match_strength': match['match_strength'],
                        'updated_at': current_time
                    })
            
            if match_data:
                # Use upsert to handle both insert and update
                result = self.supabase.table('job_match_notification').upsert(match_data).execute()
                if hasattr(result, 'error') and result.error:
                    logger.error(f"Error saving matches: {result.error}")
                else:
                    logger.info(f"Saved {len(match_data)} cosine similarity matches to database for job {job_id}")
            else:
                logger.info("No new or updated cosine similarity matches to save")
                
        except Exception as e:
            logger.error(f"Error saving cosine similarity matches to database: {e}")

    def perform_comprehensive_cosine_matching_for_job(self, job_id: str, threshold: float = 0.5, save_to_db: bool = True) -> Dict[str, Any]:
        """Perform comprehensive cosine similarity matching for one job to all applicants"""
        start_time = time.time()
        
        try:
            applicants = self.get_applicant_profiles()
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

            # Filter applicants with sufficient data
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
            
            # Calculate comprehensive cosine similarity
            cosine_scores = self.calculate_comprehensive_cosine_similarity(valid_applicants, job)
            
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
                # Safely get cosine score
                if applicant_idx >= len(cosine_scores):
                    logger.warning(f"Applicant index {applicant_idx} out of range for cosine scores")
                    continue
                    
                cosine_score = cosine_scores[applicant_idx]
                
                # Early filtering based on cosine similarity
                if cosine_score < threshold * 0.5:
                    continue
                    
                # Calculate semantic skill similarity
                skill_score = self.calculate_semantic_skill_similarity(
                    job.get('requirements', []),
                    applicant.get('skills', [])
                )

                # Calculate experience similarity
                experience_score = self.calculate_experience_similarity(job, applicant)

                # Calculate comprehensive weighted scores (now with experience)
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
            
            sorted_matches = sorted(matches, key=lambda x: x['scores']['similarity_score'], reverse=True)
            
            if save_to_db and sorted_matches:
                self.save_cosine_matches_to_db(sorted_matches, job_id)
            
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
            elif skill > 0.5:
                interpretations.append("good skill overlap")
                
            if experience > 0.7:
                interpretations.append("strong experience match")
            elif experience > 0.5:
                interpretations.append("relevant experience")
                
            return f"Match shows {', '.join(interpretations)}." if interpretations else "Basic match found."
        except Exception as e:
            logger.error(f"Error interpreting cosine match: {e}")
            return "Error in match interpretation"

    def identify_key_strengths(self, scores: Dict[str, float], job: Dict, applicant: Dict) -> List[str]:
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

    def identify_improvement_areas(self, scores: Dict[str, float], job: Dict, applicant: Dict) -> List[str]:
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

    # Batch processing methods for multiple jobs
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

    def process_single_job_matching(self, job_id: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Process cosine similarity matching for a single job"""
        try:
            logger.info(f"Processing cosine similarity matching for job: {job_id}")
            result = self.perform_comprehensive_cosine_matching_for_job(
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

    def batch_process_all_jobs(self, threshold: float = 0.5, max_jobs: int = None) -> Dict[str, Any]:
        """Process cosine similarity matching for all job postings in batches"""
        start_time = time.time()
        
        try:
            # Get all job postings
            jobs = self.get_all_job_postings(limit=max_jobs or 100)
            
            if not jobs:
                return {
                    'success': False,
                    'message': 'No job postings found',
                    'total_processed': 0,
                    'total_matches': 0
                }
            
            logger.info(f"Starting batch cosine similarity matching for {len(jobs)} jobs")
            
            # Process jobs in batches with threading
            results = []
            total_matches = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_job = {
                    executor.submit(self.process_single_job_matching, job['id'], threshold): job['id']
                    for job in jobs
                }
                
                # Process completed tasks
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
            
            # Analyze results
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

# Initialize the matcher
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    matcher = JobApplicantMatcher(SUPABASE_URL, SUPABASE_KEY, 'thenlper/gte-large')
    logger.info("Cosine similarity matcher initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize cosine similarity matcher: {e}")
    matcher = None

# API Routes with Cosine Similarity Focus - ONE JOB TO ALL APPLICANTS

@app.route('/api/semantic-matching/job/<job_id>', methods=['POST'])
def semantic_matching_for_job(job_id):
    """Perform comprehensive cosine similarity matching between one job and all applicants"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        data = request.get_json() or {}
        threshold = float(data.get('threshold', 0.5))
        top_n = int(data.get('top_n', 20))
        save_to_db = data.get('save_to_db', True)
        
        start_time = time.time()
        matching_result = matcher.perform_comprehensive_cosine_matching_for_job(job_id, threshold, save_to_db)
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

@app.route('/api/job-stats', methods=['GET'])
def get_job_stats():
    """Get statistics about job postings and their matching status"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        # Get total jobs
        jobs_response = matcher.supabase.table('jobpost') \
            .select('id, created_at', count='exact') \
            .execute()
        
        # Get active jobs
        active_jobs_response = matcher.supabase.table('jobpost') \
            .select('id', count='exact') \
            .eq('status', 'Open') \
            .eq('approval_status', 'approved') \
            .execute()
        
        # Get jobs with matches
        matches_response = matcher.supabase.table('job_match_notification') \
            .select('job_id', count='exact') \
            .execute()
        
        # Get unique jobs with matches
        unique_jobs_with_matches = set()
        if matches_response.data:
            for match in matches_response.data:
                unique_jobs_with_matches.add(match['job_id'])
        
        total_jobs = jobs_response.count if hasattr(jobs_response, 'count') else len(jobs_response.data)
        active_jobs = active_jobs_response.count if hasattr(active_jobs_response, 'count') else len(active_jobs_response.data)
        jobs_with_matches = len(unique_jobs_with_matches)
        jobs_without_matches = active_jobs - jobs_with_matches
        
        # Get recent jobs (last 7 days)
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
                'recent_jobs_last_7_days': recent_jobs
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting job stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/job-matches/<job_id>', methods=['GET'])
def get_job_matches(job_id):
    """Get matches for a specific job"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Cosine similarity matcher not initialized'}), 500
    
    try:
        matches_response = matcher.supabase.table('job_match_notification') \
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
            'matching_type': 'cosine_similarity'
        })
    except Exception as e:
        logger.error(f"Error getting job matches for {job_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'matcher_initialized': matcher is not None,
        'matching_engine': 'comprehensive_cosine_similarity',
        'matching_direction': 'one_job_to_all_applicants'
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)