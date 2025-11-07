from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict
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

# Filter out the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

class JobApplicantMatcher:
    def __init__(self, supabase_url: str, supabase_key: str, model_name: str = 'thenlper/gte-large'):
        """Initialize with Supabase connection and optimized model"""
        try:
            if not supabase_url or not supabase_key:
                raise ValueError("Supabase URL and Key must be provided")
                
            self.supabase: Client = create_client(supabase_url, supabase_key)
            
            # Initialize model with optimized settings to avoid warnings
            self.model = SentenceTransformer(
                model_name,
                device='cpu',  # Explicitly set device
                use_auth_token=False
            )
            
            # Configure model for better performance and to avoid warnings
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            self.job_cache = {}
            self.profile_cache = {}
            self.applicant_cache = {}
            self.embedding_cache = {}
            self.cache_ttl = timedelta(minutes=30)
            
            logger.info(f"Initialized JobApplicantMatcher with model: {model_name}")
            
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

    # ===== ONE APPLICANT TO MANY JOBS =====

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
            return response.data
        except Exception as e:
            logger.error(f"Error fetching job postings: {e}")
            return []

    def create_applicant_embedding_text(self, profile: Dict) -> str:
        """Create optimized text for embedding from applicant profile"""
        skills = self._normalize_skills(profile.get('skills', []) or [])
        
        # More structured and relevant text
        parts = [
            f"Position: {profile.get('position', '')}",
            f"Company: {profile.get('company', '')}",
            f"Description: {self._clean_text(profile.get('description', ''))}",
            f"Skills: {', '.join(skills[:15])}",  # Limit to top skills
            f"Experience: {profile.get('experience', 0)} years",
            f"Industry: {profile.get('industry', '')}"
        ]
        
        return " | ".join([p for p in parts if p.split(': ')[1].strip()])

    def create_job_embedding_text(self, job: Dict) -> str:
        """Create optimized text for embedding from job posting"""
        requirements = self._normalize_skills(job.get('requirements', []) or [])
        
        parts = [
            f"Title: {job.get('title', '')}",
            f"Company: {job.get('company_name', '')}",
            f"Description: {self._clean_text(job.get('description', ''))}",
            f"Requirements: {', '.join(requirements[:12])}",
            f"Job Type: {job.get('job_type', '')}",
            f"Experience Required: {job.get('experience_required', '')}"
        ]
        
        return " | ".join([p for p in parts if p.split(': ')[1].strip()])

    def _clean_text(self, text: str) -> str:
        """Clean and truncate text for better embeddings"""
        if not text:
            return ""
        # Remove extra whitespace and limit length
        cleaned = re.sub(r'\s+', ' ', text.strip())
        return cleaned[:300]  # Reasonable limit

    def _normalize_skills(self, skills) -> List[str]:
        """Normalize skills to consistent format"""
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
                if clean_skill and clean_skill not in seen:
                    normalized.append(clean_skill)
                    seen.add(clean_skill)
        return normalized

    def batch_encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Optimized batch encoding with caching"""
        cache_key = hashlib.md5("|".join(texts).encode()).hexdigest()
        now = datetime.now()
        
        if cache_key in self.embedding_cache:
            data, timestamp = self.embedding_cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return data
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Use context manager to suppress warnings during inference
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    convert_to_tensor=True, 
                    normalize_embeddings=True,
                    show_progress_bar=False,  # Faster without progress bar
                    batch_size=batch_size
                )
            embeddings.append(batch_embeddings)
        
        if embeddings:
            result = torch.cat(embeddings, dim=0)
            self.embedding_cache[cache_key] = (result, now)
            return result
        else:
            return torch.tensor([])

    def calculate_semantic_similarity_applicant_to_jobs(self, jobs: List[Dict], profile: Dict) -> np.ndarray:
        """Calculate semantic similarity between one applicant and many jobs"""
        job_texts = [self.create_job_embedding_text(job) for job in jobs]
        profile_text = self.create_applicant_embedding_text(profile)
        
        logger.info("Generating job embeddings...")
        job_embeddings = self.batch_encode_texts(job_texts, batch_size=64)
        
        logger.info("Generating applicant embedding...")
        with torch.no_grad():
            profile_embedding = self.model.encode(
                [profile_text], 
                convert_to_tensor=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
        
        logger.info("Calculating cosine similarity...")
        similarity_scores = util.cos_sim(job_embeddings, profile_embedding)
        
        return similarity_scores.flatten()

    def calculate_skill_similarity(self, job_requirements: List[str], applicant_skills: List[str]) -> float:
        """Calculate skill similarity using Jaccard similarity for speed and accuracy"""
        if not job_requirements or not applicant_skills:
            return 0.0
        
        job_reqs = set(self._normalize_skills(job_requirements))
        applicant_skills_set = set(self._normalize_skills(applicant_skills))
        
        if not job_reqs:
            return 0.0
        
        # Jaccard similarity
        intersection = len(job_reqs.intersection(applicant_skills_set))
        union = len(job_reqs.union(applicant_skills_set))
        
        if union == 0:
            return 0.0
            
        similarity = intersection / union
        
        # Boost score if applicant has most required skills
        coverage = intersection / len(job_reqs) if job_reqs else 0
        return max(similarity, coverage * 0.8)  # Blend both metrics

    def calculate_experience_similarity(self, job_experience: str, applicant_experience: float) -> float:
        """Calculate experience similarity with accurate 5-point weighting (0.5)"""
        if not job_experience or applicant_experience is None:
            return 0.7  # Neutral score when no requirement
        
        job_years = self._extract_experience_years(job_experience)
        
        if job_years == 0:  # No specific requirement
            return 0.8
            
        applicant_exp = applicant_experience or 0
        
        # More accurate experience matching
        if applicant_exp >= job_years:
            return 1.0  # Meets or exceeds requirement
        elif applicant_exp >= job_years * 0.8:
            return 0.9
        elif applicant_exp >= job_years * 0.6:
            return 0.7
        elif applicant_exp >= job_years * 0.4:
            return 0.5
        else:
            return 0.3

    def _extract_experience_years(self, experience_text: str) -> float:
        """Accurately extract experience years from text"""
        if not experience_text:
            return 0
            
        text_lower = str(experience_text).lower()
        
        # Look for explicit year patterns
        year_matches = re.findall(r'(\d+)\s*(?:year|yr|years|yrs)', text_lower)
        if year_matches:
            return float(year_matches[0])
        
        # Look for range patterns
        range_matches = re.findall(r'(\d+)\s*-\s*(\d+)', text_lower)
        if range_matches:
            return (float(range_matches[0][0]) + float(range_matches[0][1])) / 2
        
        # Level-based mapping
        level_mapping = {
            'entry': 1.0, 'junior': 2.0, 
            'mid': 3.0, 'intermediate': 3.0, 'experienced': 4.0,
            'senior': 5.0, 'lead': 6.0, 'principal': 7.0,
            'executive': 10.0, 'director': 8.0, 'vp': 9.0, 'manager': 5.0
        }
        
        for level, years in level_mapping.items():
            if level in text_lower:
                return years
                
        return 2.0  # Default assumption

    def calculate_weighted_score(self, semantic_score: float, skill_score: float, experience_score: float) -> float:
        """Calculate weighted score with experience weighted at 0.5"""
        return (0.75 * semantic_score) + (0.20 * skill_score) + (0.05 * experience_score)

    def get_existing_matches_for_applicant(self, user_id: str) -> Dict[str, Dict]:
        """Get existing matches for user to prevent duplicates"""
        try:
            response = self.supabase.table('job_matches') \
                .select('job_id, similarity_score, match_strength, updated_at') \
                .eq('applicant_id', user_id) \
                .execute()
            
            existing_matches = {}
            for match in response.data:
                existing_matches[match['job_id']] = match
            return existing_matches
        except Exception as e:
            logger.error(f"Error fetching existing matches: {e}")
            return {}

    def save_applicant_matches_to_db(self, matches: List[Dict], user_id: str):
        """Save matches with duplicate prevention"""
        if not matches:
            return
            
        try:
            # Get existing matches to avoid unnecessary updates
            existing_matches = self.get_existing_matches_for_applicant(user_id)
            
            # Prepare data for upsert, only including matches that are new or updated
            match_data = []
            current_time = datetime.now().isoformat()
            
            for match in matches:
                job_id = match['job_id']
                existing_match = existing_matches.get(job_id)
                
                # Only include if it's a new match or score has significantly changed
                if (not existing_match or 
                    abs(existing_match['similarity_score'] - match['score']) > 0.01):  # 1% threshold for update
                    
                    match_data.append({
                        'applicant_id': user_id,
                        'job_id': job_id,
                        'similarity_score': match['score'],
                        'match_strength': match['match_strength'],
                        'updated_at': current_time
                    })
            
            if match_data:
                # Single batch upsert
                self.supabase.table('job_matches').upsert(match_data).execute()
                logger.info(f"Saved/updated {len(match_data)} matches to database (filtered from {len(matches)} calculated matches)")
            else:
                logger.info("No new or updated matches to save")
                
        except Exception as e:
            logger.error(f"Error saving matches to database: {e}")

    def perform_semantic_matching_for_user(self, user_id: str, threshold: float = 0.5, save_to_db: bool = True) -> List[Dict]:
        """Optimized semantic matching between one applicant and many jobs"""
        start_time = time.time()
        
        jobs = self.get_job_postings()
        profile = self.get_applicant_profile(user_id)
        
        if not jobs or not profile:
            logger.warning("No jobs or profile found for matching")
            return []
        
        # Calculate semantic similarity in batch
        semantic_scores = self.calculate_semantic_similarity_applicant_to_jobs(jobs, profile)
        
        matches = []
        for job_idx, job in enumerate(jobs):
            semantic_score = semantic_scores[job_idx].item()
            
            # Early filtering - skip if semantic score is too low
            if semantic_score < threshold * 0.6:
                continue
                
            skill_score = self.calculate_skill_similarity(
                job.get('requirements', []),
                profile.get('skills', [])
            )
            
            experience_score = self.calculate_experience_similarity(
                job.get('experience_required', ''),
                profile.get('experience', 0)
            )
            
            weighted_score = self.calculate_weighted_score(semantic_score, skill_score, experience_score)
            
            if weighted_score >= threshold:
                match_strength = self.get_match_strength(weighted_score)
                
                match_data = {
                    'job_id': job['id'],
                    'applicant_id': user_id,
                    'score': float(weighted_score),
                    'job_title': job['title'],
                    'job_company': job['company_name'],
                    'match_strength': match_strength,
                    'semantic_score': semantic_score,
                    'skill_score': skill_score,
                    'experience_score': experience_score
                }
                matches.append(match_data)
        
        sorted_matches = sorted(matches, key=lambda x: x['score'], reverse=True)
        
        # Only save to database if explicitly requested
        if save_to_db and sorted_matches:
            self.save_applicant_matches_to_db(sorted_matches, user_id)
        
        logger.info(f"Matching completed in {time.time() - start_time:.2f}s. Found {len(sorted_matches)} matches.")
        return sorted_matches

    # ===== ONE JOB TO MANY APPLICANTS =====

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

    def get_applicant_profiles(self) -> List[Dict]:
        """Fetch all applicant profiles with caching"""
        cache_key = "all_applicants"
        now = datetime.now()
        
        if cache_key in self.applicant_cache:
            data, timestamp = self.applicant_cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return data
        
        try:
            response = self.supabase.table('applicant_profiles') \
                .select('*') \
                .execute()
            logger.info(f"Fetched {len(response.data)} applicant profiles")
            self.applicant_cache[cache_key] = (response.data, now)
            return response.data
        except Exception as e:
            logger.error(f"Error fetching applicant profiles: {e}")
            return []

    def calculate_semantic_similarity_job_to_applicants(self, applicants: List[Dict], job: Dict) -> np.ndarray:
        """Calculate semantic similarity between one job and many applicants"""
        applicant_texts = [self.create_applicant_embedding_text(applicant) for applicant in applicants]
        job_text = self.create_job_embedding_text(job)
        
        logger.info("Generating applicant embeddings...")
        applicant_embeddings = self.batch_encode_texts(applicant_texts, batch_size=64)
        
        logger.info("Generating job embedding...")
        with torch.no_grad():
            job_embedding = self.model.encode(
                [job_text], 
                convert_to_tensor=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
        
        logger.info("Calculating cosine similarity...")
        similarity_scores = util.cos_sim(applicant_embeddings, job_embedding)
        
        return similarity_scores.flatten()

    def get_existing_matches_for_job(self, job_id: str) -> Dict[str, Dict]:
        """Get existing matches for a job to prevent duplicates"""
        try:
            response = self.supabase.table('applicant_match') \
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

    def save_job_matches_to_db(self, matches: List[Dict], job_id: str):
        """Save matches with duplicate prevention"""
        if not matches:
            return
            
        try:
            # Get existing matches to avoid unnecessary updates
            existing_matches = self.get_existing_matches_for_job(job_id)
            
            # Prepare data for upsert, only including matches that are new or updated
            match_data = []
            current_time = datetime.now().isoformat()
            
            for match in matches:
                applicant_id = match['applicant_id']
                existing_match = existing_matches.get(applicant_id)
                
                # Only include if it's a new match or score has significantly changed
                if (not existing_match or 
                    abs(existing_match['similarity_score'] - match['score']) > 0.01):  # 1% threshold for update
                    
                    match_data.append({
                        'applicant_id': applicant_id,
                        'job_id': job_id,
                        'similarity_score': match['score'],
                        'match_strength': match['match_strength'],
                        'updated_at': current_time
                    })
            
            if match_data:
                # Single batch upsert
                self.supabase.table('applicant_match').upsert(match_data).execute()
                logger.info(f"Saved/updated {len(match_data)} matches to database for job {job_id} (filtered from {len(matches)} calculated matches)")
            else:
                logger.info(f"No new or updated matches to save for job {job_id}")
                
        except Exception as e:
            logger.error(f"Error saving matches to database for job {job_id}: {e}")

    def perform_semantic_matching_for_job(self, job_id: str, threshold: float = 0.5, save_to_db: bool = True) -> List[Dict]:
        """Optimized semantic matching between one job and many applicants"""
        start_time = time.time()
        
        applicants = self.get_applicant_profiles()
        job = self.get_job_profile(job_id)
        
        if not applicants or not job:
            logger.warning("No applicants or job found for matching")
            return []
        
        # Calculate semantic similarity in batch
        semantic_scores = self.calculate_semantic_similarity_job_to_applicants(applicants, job)
        
        matches = []
        for applicant_idx, applicant in enumerate(applicants):
            semantic_score = semantic_scores[applicant_idx].item()
            
            # Early filtering - skip if semantic score is too low
            if semantic_score < threshold * 0.6:
                continue
                
            skill_score = self.calculate_skill_similarity(
                job.get('requirements', []),
                applicant.get('skills', [])
            )
            
            experience_score = self.calculate_experience_similarity(
                job.get('experience_required', ''),
                applicant.get('experience', 0)
            )
            
            weighted_score = self.calculate_weighted_score(semantic_score, skill_score, experience_score)
            
            if weighted_score >= threshold:
                match_strength = self.get_match_strength(weighted_score)
                
                match_data = {
                    'applicant_id': applicant['id'],
                    'job_id': job_id,
                    'score': float(weighted_score),
                    'applicant_name': f"{applicant.get('first_name', '')} {applicant.get('last_name', '')}".strip(),
                    'applicant_position': applicant.get('position', ''),
                    'match_strength': match_strength,
                    'semantic_score': semantic_score,
                    'skill_score': skill_score,
                    'experience_score': experience_score
                }
                matches.append(match_data)
        
        sorted_matches = sorted(matches, key=lambda x: x['score'], reverse=True)
        
        # Only save to database if explicitly requested
        if save_to_db and sorted_matches:
            self.save_job_matches_to_db(sorted_matches, job_id)
        
        logger.info(f"Matching completed in {time.time() - start_time:.2f}s. Found {len(sorted_matches)} matches for job {job_id}.")
        return sorted_matches

    def get_match_strength(self, score: float) -> str:
        """Determine match strength based on score"""
        percentage = score * 100
        
        if percentage >= 80:
            return 'Strong'
        elif percentage >= 65:
            return 'Good'
        elif percentage >= 50:
            return 'Moderate'
        else:
            return 'Weak'

    def get_top_matches(self, matches: List[Dict], top_n: int = 10) -> List[Dict]:
        """Get top N matches by score"""
        return matches[:top_n]

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    matcher = JobApplicantMatcher(SUPABASE_URL, SUPABASE_KEY, 'thenlper/gte-large')
except Exception as e:
    logger.error(f"Failed to initialize matcher: {e}")
    matcher = None

# ===== API ROUTES FOR ONE APPLICANT TO MANY JOBS =====

@app.route('/api/semantic-matching', methods=['POST'])
def semantic_matching():
    """Perform semantic matching between jobs and the current user (one applicant to many jobs)"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Matcher not initialized'}), 500
    
    try:
        user_id = request.headers.get('User-Id')
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID not provided'}), 400
        
        data = request.get_json() or {}
        threshold = float(data.get('threshold', 0.5))
        top_n = int(data.get('top_n', 20))
        save_to_db = data.get('save_to_db', True)  # Default to True for backward compatibility
        
        start_time = time.time()
        matches = matcher.perform_semantic_matching_for_user(user_id, threshold, save_to_db)
        top_matches = matcher.get_top_matches(matches, top_n)
        response_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'data': {
                'total_matches': len(matches),
                'top_matches': top_matches,
                'threshold_used': threshold,
                'response_time_seconds': round(response_time, 2),
                'saved_to_db': save_to_db
            }
        })
    except Exception as e:
        logger.error(f"Error in semantic matching: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/my-matches', methods=['GET'])
def get_my_matches():
    """Get matches for the current user"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Matcher not initialized'}), 500
    
    try:
        user_id = request.headers.get('User-Id')
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID not provided'}), 400
        
        matches_response = matcher.supabase.table('job_matches') \
            .select('''
                job_id, 
                similarity_score,
                match_strength,
                jobpost (*)
            ''') \
            .eq('applicant_id', user_id) \
            .order('similarity_score', ascending=False) \
            .execute()
        
        return jsonify({
            'success': True,
            'matches': matches_response.data
        })
    except Exception as e:
        logger.error(f"Error getting user matches: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== API ROUTES FOR ONE JOB TO MANY APPLICANTS =====

@app.route('/api/semantic-matching/job/<job_id>', methods=['POST'])
def semantic_matching_for_job(job_id):
    """Perform semantic matching between one job and many applicants"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Matcher not initialized'}), 500
    
    try:
        data = request.get_json() or {}
        threshold = float(data.get('threshold', 0.5))
        top_n = int(data.get('top_n', 20))
        save_to_db = data.get('save_to_db', True)  # Default to True for backward compatibility
        
        start_time = time.time()
        matches = matcher.perform_semantic_matching_for_job(job_id, threshold, save_to_db)
        top_matches = matcher.get_top_matches(matches, top_n)
        response_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'data': {
                'job_id': job_id,
                'total_matches': len(matches),
                'top_matches': top_matches,
                'threshold_used': threshold,
                'response_time_seconds': round(response_time, 2),
                'saved_to_db': save_to_db
            }
        })
    except Exception as e:
        logger.error(f"Error in semantic matching for job {job_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/job-matches/<job_id>', methods=['GET'])
def get_job_matches(job_id):
    """Get matches for a specific job"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Matcher not initialized'}), 500
    
    try:
        matches_response = matcher.supabase.table('applicant_match') \
            .select('''
                applicant_id, 
                similarity_score,
                match_strength,
                applicant_profiles (*)
            ''') \
            .eq('job_id', job_id) \
            .order('similarity_score', ascending=False) \
            .execute()
        
        return jsonify({
            'success': True,
            'matches': matches_response.data
        })
    except Exception as e:
        logger.error(f"Error getting job matches for {job_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== COMMON ROUTES =====

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get matching statistics"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Matcher not initialized'}), 500
    
    try:
        applicants = matcher.supabase.table('applicant_profiles').select('id', count='exact').execute()
        jobs = matcher.supabase.table('jobpost').select('id', count='exact').execute()
        job_matches = matcher.supabase.table('job_matches').select('id', count='exact').execute()
        applicant_matches = matcher.supabase.table('applicant_match').select('id', count='exact').execute()
        
        return jsonify({
            'success': True,
            'data': {
                'applicants_count': applicants.count,
                'jobs_count': jobs.count,
                'job_matches_count': job_matches.count,
                'applicant_matches_count': applicant_matches.count
            }
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'matcher_initialized': matcher is not None
    })

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear caches (admin endpoint)"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Matcher not initialized'}), 500
    
    try:
        matcher.job_cache.clear()
        matcher.profile_cache.clear()
        matcher.applicant_cache.clear()
        matcher.embedding_cache.clear()
        return jsonify({'success': True, 'message': 'Caches cleared'})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)