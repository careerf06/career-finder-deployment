#functional sa notification

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

    def calculate_batch_semantic_similarity(self, jobs: List[Dict], applicants: List[Dict]) -> np.ndarray:
        """Calculate semantic similarity between many applicants and many jobs"""
        job_texts = [self.create_job_embedding_text(job) for job in jobs]
        applicant_texts = [self.create_applicant_embedding_text(applicant) for applicant in applicants]
        
        logger.info("Generating job embeddings...")
        job_embeddings = self.batch_encode_texts(job_texts, batch_size=64)
        
        logger.info("Generating applicant embeddings...")
        applicant_embeddings = self.batch_encode_texts(applicant_texts, batch_size=64)
        
        logger.info("Calculating cosine similarity matrix...")
        similarity_matrix = util.cos_sim(applicant_embeddings, job_embeddings)
        
        return similarity_matrix.numpy()

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

    def get_existing_matches(self) -> Dict[str, Dict]:
        """Get all existing matches to prevent duplicates"""
        try:
            response = self.supabase.table('job_match_notification') \
                .select('applicant_id, job_id, similarity_score, match_strength, updated_at') \
                .execute()
            
            existing_matches = {}
            for match in response.data:
                key = f"{match['applicant_id']}_{match['job_id']}"
                existing_matches[key] = match
            return existing_matches
        except Exception as e:
            logger.error(f"Error fetching existing matches: {e}")
            return {}

    def save_batch_matches_to_db(self, matches: List[Dict]):
        """Save batch matches with duplicate prevention"""
        if not matches:
            return
            
        try:
            # Get existing matches to avoid unnecessary updates
            existing_matches = self.get_existing_matches()
            
            # Prepare data for upsert, only including matches that are new or updated
            match_data = []
            current_time = datetime.now().isoformat()
            
            for match in matches:
                match_key = f"{match['applicant_id']}_{match['job_id']}"
                existing_match = existing_matches.get(match_key)
                
                # Only include if it's a new match or score has significantly changed
                if (not existing_match or 
                    abs(existing_match['similarity_score'] - match['score']) > 0.01):  # 1% threshold for update
                    
                    match_data.append({
                        'applicant_id': match['applicant_id'],
                        'job_id': match['job_id'],
                        'similarity_score': match['score'],
                        'match_strength': match['match_strength'],
                        'updated_at': current_time
                    })
            
            if match_data:
                # Batch upsert in chunks to avoid payload size limits
                chunk_size = 100
                for i in range(0, len(match_data), chunk_size):
                    chunk = match_data[i:i + chunk_size]
                    self.supabase.table('job_match_notification').upsert(chunk).execute()
                    logger.info(f"Saved/updated chunk {i//chunk_size + 1}/{(len(match_data)-1)//chunk_size + 1}")
                
                logger.info(f"Saved/updated {len(match_data)} matches to database (filtered from {len(matches)} calculated matches)")
            else:
                logger.info("No new or updated matches to save")
                
        except Exception as e:
            logger.error(f"Error saving matches to database: {e}")

    def perform_batch_semantic_matching(self, threshold: float = 0.5, save_to_db: bool = True) -> Dict[str, List[Dict]]:
        """Optimized batch semantic matching between many applicants and many jobs"""
        start_time = time.time()
        
        applicants = self.get_applicant_profiles()
        jobs = self.get_job_postings()
        
        if not applicants or not jobs:
            logger.warning("No applicants or jobs found for matching")
            return {}
        
        logger.info(f"Starting batch matching for {len(applicants)} applicants and {len(jobs)} jobs")
        
        # Calculate semantic similarity matrix
        similarity_matrix = self.calculate_batch_semantic_similarity(jobs, applicants)
        
        all_matches = []
        applicant_matches = {}
        
        # Process each applicant-job combination
        for applicant_idx, applicant in enumerate(applicants):
            applicant_id = applicant['id']
            applicant_matches[applicant_id] = []
            
            for job_idx, job in enumerate(jobs):
                semantic_score = similarity_matrix[applicant_idx, job_idx]
                
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
                        'applicant_id': applicant_id,
                        'job_id': job['id'],
                        'score': float(weighted_score),
                        'applicant_name': f"{applicant.get('first_name', '')} {applicant.get('last_name', '')}".strip(),
                        'applicant_position': applicant.get('position', ''),
                        'job_title': job['title'],
                        'job_company': job['company_name'],
                        'match_strength': match_strength,
                        'semantic_score': float(semantic_score),
                        'skill_score': skill_score,
                        'experience_score': experience_score
                    }
                    all_matches.append(match_data)
                    applicant_matches[applicant_id].append(match_data)
        
        # Sort matches for each applicant
        for applicant_id in applicant_matches:
            applicant_matches[applicant_id] = sorted(applicant_matches[applicant_id], key=lambda x: x['score'], reverse=True)
        
        # Only save to database if explicitly requested
        if save_to_db and all_matches:
            self.save_batch_matches_to_db(all_matches)
        
        total_matches = len(all_matches)
        logger.info(f"Batch matching completed in {time.time() - start_time:.2f}s. Found {total_matches} total matches across {len(applicants)} applicants.")
        
        return {
            'total_matches': total_matches,
            'applicants_processed': len(applicants),
            'jobs_processed': len(jobs),
            'matches_by_applicant': applicant_matches
        }

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

@app.route('/api/semantic-matching/batch', methods=['POST'])
def batch_semantic_matching():
    """Perform batch semantic matching between all applicants and all jobs"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Matcher not initialized'}), 500
    
    try:
        data = request.get_json() or {}
        threshold = float(data.get('threshold', 0.5))
        save_to_db = data.get('save_to_db', True)
        
        start_time = time.time()
        result = matcher.perform_batch_semantic_matching(threshold, save_to_db)
        response_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'data': {
                **result,
                'threshold_used': threshold,
                'response_time_seconds': round(response_time, 2),
                'saved_to_db': save_to_db
            }
        })
    except Exception as e:
        logger.error(f"Error in batch semantic matching: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/semantic-matching', methods=['POST'])
def semantic_matching():
    """Perform semantic matching between jobs and the current user (single applicant)"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Matcher not initialized'}), 500
    
    try:
        user_id = request.headers.get('User-Id')
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID not provided'}), 400
        
        data = request.get_json() or {}
        threshold = float(data.get('threshold', 0.5))
        top_n = int(data.get('top_n', 20))
        save_to_db = data.get('save_to_db', True)
        
        # Use the batch matching but filter for single user
        batch_result = matcher.perform_batch_semantic_matching(threshold, save_to_db)
        user_matches = batch_result['matches_by_applicant'].get(user_id, [])
        top_matches = matcher.get_top_matches(user_matches, top_n)
        
        return jsonify({
            'success': True,
            'data': {
                'total_matches': len(user_matches),
                'top_matches': top_matches,
                'threshold_used': threshold,
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
        
        matches_response = matcher.supabase.table('job_match_notification') \
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

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get matching statistics"""
    if not matcher:
        return jsonify({'success': False, 'error': 'Matcher not initialized'}), 500
    
    try:
        applicants = matcher.supabase.table('applicant_profiles').select('id', count='exact').execute()
        jobs = matcher.supabase.table('jobpost').select('id', count='exact').execute()
        matches = matcher.supabase.table('job_match_notification').select('id', count='exact').execute()
        
        return jsonify({
            'success': True,
            'data': {
                'applicants_count': applicants.count,
                'jobs_count': jobs.count,
                'matches_count': matches.count
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
        matcher.applicant_cache.clear()
        matcher.embedding_cache.clear()
        return jsonify({'success': True, 'message': 'Caches cleared'})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)