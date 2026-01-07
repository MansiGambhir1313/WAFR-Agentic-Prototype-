"""
AWS Well-Architected Tool API Client
Provides programmatic access to WA Tool for autonomous WAFR operations
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class WellArchitectedToolClient:
    """Client for interacting with AWS Well-Architected Tool API."""
    
    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize WA Tool client with optimized connection pool.
        
        Args:
            region: AWS region (default: us-east-1)
        """
        self.region = region
        
        # Configure boto3 client with larger connection pool and adaptive retries
        config = Config(
            max_pool_connections=20,  # Increase from default 10
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'  # Adaptive retry mode for better handling
            },
            connect_timeout=60,
            read_timeout=120  # Increased read timeout
        )
        
        self.client = boto3.client('wellarchitected', region_name=region, config=config)
        self._review_owner = None  # Cache for review owner
    
    def _get_review_owner(self) -> str:
        """
        Get the review owner email from AWS credentials.
        
        Returns:
            Email address of the current AWS user
        """
        if self._review_owner:
            return self._review_owner
        
        try:
            # Try to get current user identity
            sts_client = boto3.client('sts', region_name=self.region)
            identity = sts_client.get_caller_identity()
            arn = identity.get('Arn', '')
            
            # Extract email from ARN or use account ID
            # ARN format: arn:aws:iam::account-id:user/username
            if 'user/' in arn:
                username = arn.split('user/')[-1]
                # If it looks like an email, use it; otherwise construct one
                if '@' in username:
                    self._review_owner = username
                else:
                    # Use account ID as fallback (will need manual update)
                    account_id = identity.get('Account', 'unknown')
                    self._review_owner = f"wafr-{account_id}@aws.local"
            else:
                # For roles, use account ID
                account_id = identity.get('Account', 'unknown')
                self._review_owner = f"wafr-{account_id}@aws.local"
            
            logger.info(f"Using ReviewOwner: {self._review_owner}")
            return self._review_owner
            
        except Exception as e:
            logger.warning(f"Could not determine review owner from AWS identity: {e}")
            # Fallback to a default
            self._review_owner = "wafr-automation@aws.local"
            return self._review_owner
    
    def create_workload(
        self,
        workload_name: str,
        description: str,
        environment: str = 'PRODUCTION',
        aws_regions: Optional[List[str]] = None,
        lenses: Optional[List[str]] = None,
        review_owner: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new workload in WA Tool.
        
        Args:
            workload_name: Name of the workload
            description: Description of the workload
            environment: Environment type (PRODUCTION, PREPRODUCTION, DEVELOPMENT)
            aws_regions: List of AWS regions
            lenses: List of lens aliases (e.g., ['wellarchitected', 'genai'])
            review_owner: Email of the review owner (if not provided, will be auto-detected)
            tags: Tags for the workload
            
        Returns:
            Workload creation response
        """
        try:
            # Normalize lens aliases
            from agents.lens_manager import LensManager
            
            if not lenses:
                lenses = ['wellarchitected']  # Default: just wellarchitected
            else:
                # Normalize all lens aliases
                normalized_lenses = []
                for lens in lenses:
                    normalized = LensManager.normalize_lens_alias(lens)
                    if normalized not in normalized_lenses:
                        normalized_lenses.append(normalized)
                lenses = normalized_lenses
                
                # Ensure wellarchitected is always included
                if 'wellarchitected' not in lenses:
                    lenses = ['wellarchitected'] + lenses
            
            if not aws_regions:
                aws_regions = [self.region]
            
            # ReviewOwner is required - get it if not provided
            if not review_owner:
                review_owner = self._get_review_owner()
            
            params = {
                'WorkloadName': workload_name,
                'Description': description,
                'Environment': environment,
                'AwsRegions': aws_regions,
                'Lenses': lenses,
                'ReviewOwner': review_owner,  # Required parameter
                'ClientRequestToken': f"wafr-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            }
            
            if tags:
                params['Tags'] = tags
            
            try:
                response = self.client.create_workload(**params)
                logger.info(f"Created workload: {workload_name} (ID: {response.get('WorkloadId')}) with lenses: {lenses}")
                return response
            except ClientError as e:
                # Handle lens access errors gracefully
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', '')
                
                # Check if it's a lens access error
                if error_code == 'ValidationException' and ('not authorized' in error_message.lower() or 'Failed to get lenses' in error_message):
                    logger.warning(f"Lens access error: {error_message}")
                    
                    # Try to identify which lens failed
                    failed_lenses = []
                    for lens in lenses:
                        if lens.lower() in error_message.lower():
                            failed_lenses.append(lens)
                    
                    # Remove failed lenses and retry
                    working_lenses = [l for l in lenses if l not in failed_lenses]
                    
                    if not working_lenses:
                        # If all lenses failed, use wellarchitected as fallback
                        working_lenses = ['wellarchitected']
                        logger.warning("All requested lenses failed, falling back to wellarchitected only")
                    else:
                        logger.info(f"Retrying with accessible lenses: {working_lenses} (removed: {failed_lenses})")
                    
                    # Retry with working lenses only
                    params['Lenses'] = working_lenses
                    params['ClientRequestToken'] = f"wafr-{datetime.now().strftime('%Y%m%d%H%M%S')}"  # New token for retry
                    
                    response = self.client.create_workload(**params)
                    logger.info(f"Created workload: {workload_name} (ID: {response.get('WorkloadId')}) with lenses: {working_lenses}")
                    
                    # Add metadata about skipped lenses
                    if failed_lenses:
                        response['_metadata'] = {
                            'requested_lenses': lenses,
                            'working_lenses': working_lenses,
                            'skipped_lenses': failed_lenses,
                            'skip_reason': 'lens_access_denied'
                        }
                    
                    return response
                else:
                    # Re-raise if it's a different error
                    raise
            
        except ClientError as e:
            logger.error(f"Error creating workload: {str(e)}")
            raise
    
    def get_workload(self, workload_id: str) -> Dict[str, Any]:
        """Get workload details."""
        try:
            response = self.client.get_workload(WorkloadId=workload_id)
            return response
        except ClientError as e:
            logger.error(f"Error getting workload: {str(e)}")
            raise
    
    def list_workloads(self) -> List[Dict[str, Any]]:
        """List all workloads."""
        try:
            workloads = []
            # Try direct call first (list_workloads doesn't support pagination)
            response = self.client.list_workloads()
            workloads = response.get('WorkloadSummaries', [])
            return workloads
        except ClientError as e:
            logger.error(f"Error listing workloads: {str(e)}")
            return []
    
    def create_milestone(
        self,
        workload_id: str,
        milestone_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a milestone for a workload.
        
        Args:
            workload_id: Workload ID
            milestone_name: Optional milestone name (default: timestamp-based)
            
        Returns:
            Milestone creation response
        """
        try:
            if not milestone_name:
                milestone_name = f"Review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            response = self.client.create_milestone(
                WorkloadId=workload_id,
                MilestoneName=milestone_name
            )
            logger.info(f"Created milestone: {milestone_name} for workload {workload_id}")
            return response
            
        except ClientError as e:
            logger.error(f"Error creating milestone: {str(e)}")
            raise
    
    def get_answer(
        self,
        workload_id: str,
        lens_alias: str,
        question_id: str,
        milestone_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get answer for a specific question.
        
        Args:
            workload_id: Workload ID
            lens_alias: Lens alias (e.g., 'wellarchitected')
            question_id: Question ID (e.g., 'OPS_01')
            milestone_number: Optional milestone number
            
        Returns:
            Answer details
        """
        try:
            params = {
                'WorkloadId': workload_id,
                'LensAlias': lens_alias,
                'QuestionId': question_id
            }
            
            if milestone_number:
                params['MilestoneNumber'] = milestone_number
            
            response = self.client.get_answer(**params)
            return response
            
        except ClientError as e:
            logger.error(f"Error getting answer: {str(e)}")
            raise
    
    def update_answer(
        self,
        workload_id: str,
        lens_alias: str,
        question_id: str,
        selected_choices: List[str],
        notes: Optional[str] = None,
        is_applicable: bool = True,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update answer for a question.
        
        Args:
            workload_id: Workload ID
            lens_alias: Lens alias
            question_id: Question ID
            selected_choices: List of selected choice IDs
            notes: Optional notes
            is_applicable: Whether question is applicable
            reason: Optional reason for not applicable
            
        Returns:
            Update response
        """
        try:
            params = {
                'WorkloadId': workload_id,
                'LensAlias': lens_alias,
                'QuestionId': question_id,
                'SelectedChoices': selected_choices
            }
            
            if notes:
                params['Notes'] = notes
            
            if not is_applicable:
                params['IsApplicable'] = False
                if reason:
                    params['Reason'] = reason
            
            response = self.client.update_answer(**params)
            logger.info(f"Updated answer for {question_id} in workload {workload_id}")
            return response
            
        except ClientError as e:
            logger.error(f"Error updating answer: {str(e)}")
            raise
    
    def list_answers(
        self,
        workload_id: str,
        lens_alias: str,
        milestone_number: Optional[int] = None,
        pillar_id: Optional[str] = None,
        question_priority: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all answers for a workload.
        
        Args:
            workload_id: Workload ID
            lens_alias: Lens alias
            milestone_number: Optional milestone number
            pillar_id: Optional pillar filter
            question_priority: Optional priority filter
            
        Returns:
            List of answers
        """
        try:
            params = {
                'WorkloadId': workload_id,
                'LensAlias': lens_alias
            }
            
            if milestone_number:
                params['MilestoneNumber'] = milestone_number
            if pillar_id:
                params['PillarId'] = pillar_id
            if question_priority:
                params['QuestionPriority'] = question_priority
            
            # list_answers doesn't support pagination, call directly
            response = self.client.list_answers(**params)
            answers = response.get('AnswerSummaries', [])
            
            return answers
            
        except ClientError as e:
            logger.error(f"Error listing answers: {str(e)}")
            return []
    
    def get_lens_review(
        self,
        workload_id: str,
        lens_alias: str,
        milestone_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get lens review for a workload.
        
        Args:
            workload_id: Workload ID
            lens_alias: Lens alias
            milestone_number: Optional milestone number
            
        Returns:
            Lens review details
        """
        try:
            params = {
                'WorkloadId': workload_id,
                'LensAlias': lens_alias
            }
            
            if milestone_number:
                params['MilestoneNumber'] = milestone_number
            
            response = self.client.get_lens_review(**params)
            return response
            
        except ClientError as e:
            logger.error(f"Error getting lens review: {str(e)}")
            raise
    
    def get_lens_review_report(
        self,
        workload_id: str,
        lens_alias: str,
        milestone_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get lens review report (official AWS WAFR PDF).
        
        Args:
            workload_id: Workload ID
            lens_alias: Lens alias
            milestone_number: Optional milestone number
            
        Returns:
            Review report with Base64-encoded PDF
        """
        try:
            params = {
                'WorkloadId': workload_id,
                'LensAlias': lens_alias
            }
            
            if milestone_number:
                params['MilestoneNumber'] = milestone_number
            
            response = self.client.get_lens_review_report(**params)
            return response
            
        except ClientError as e:
            logger.error(f"Error getting lens review report: {str(e)}")
            raise
    
    def list_lens_review_improvements(
        self,
        workload_id: str,
        lens_alias: str,
        pillar_id: Optional[str] = None,
        milestone_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List improvement items (HRIs and MRIs) for a workload.
        
        Args:
            workload_id: Workload ID
            lens_alias: Lens alias
            pillar_id: Optional pillar filter
            milestone_number: Optional milestone number
            
        Returns:
            List of improvement summaries
        """
        try:
            params = {
                'WorkloadId': workload_id,
                'LensAlias': lens_alias
            }
            
            if pillar_id:
                params['PillarId'] = pillar_id
            if milestone_number:
                params['MilestoneNumber'] = milestone_number
            
            improvements = []
            paginator = self.client.get_paginator('list_lens_review_improvements')
            
            for page in paginator.paginate(**params):
                improvements.extend(page.get('ImprovementSummaries', []))
            
            return improvements
            
        except ClientError as e:
            logger.error(f"Error listing improvements: {str(e)}")
            return []
    
    def get_consolidated_report(
        self,
        workload_ids: List[str],
        format: str = 'PDF'
    ) -> Dict[str, Any]:
        """
        Get consolidated report for multiple workloads.
        
        Args:
            workload_ids: List of workload IDs
            format: Report format (PDF, JSON)
            
        Returns:
            Consolidated report
        """
        try:
            response = self.client.get_consolidated_report(
                WorkloadIds=workload_ids,
                Format=format
            )
            return response
            
        except ClientError as e:
            logger.error(f"Error getting consolidated report: {str(e)}")
            raise
    
    def list_lenses(self) -> List[Dict[str, Any]]:
        """List available lenses."""
        try:
            lenses = []
            # Try direct call first (list_lenses doesn't support pagination)
            response = self.client.list_lenses()
            lenses = response.get('LensSummaries', [])
            return lenses
        except ClientError as e:
            logger.error(f"Error listing lenses: {str(e)}")
            return []
    
    def get_lens(self, lens_alias: str, lens_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get lens details.
        
        Args:
            lens_alias: Lens alias
            lens_version: Optional lens version
            
        Returns:
            Lens details
        """
        try:
            params = {'LensAlias': lens_alias}
            if lens_version:
                params['LensVersion'] = lens_version
            
            response = self.client.get_lens(**params)
            return response
            
        except ClientError as e:
            logger.error(f"Error getting lens: {str(e)}")
            raise
    
    def update_workload(
        self,
        workload_id: str,
        workload_name: Optional[str] = None,
        description: Optional[str] = None,
        environment: Optional[str] = None,
        aws_regions: Optional[List[str]] = None,
        review_owner: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update workload details.
        
        Args:
            workload_id: Workload ID
            workload_name: Optional new name
            description: Optional new description
            environment: Optional new environment
            aws_regions: Optional new regions
            review_owner: Optional new review owner
            
        Returns:
            Update response
        """
        try:
            params = {'WorkloadId': workload_id}
            
            if workload_name:
                params['WorkloadName'] = workload_name
            if description:
                params['Description'] = description
            if environment:
                params['Environment'] = environment
            if aws_regions:
                params['AwsRegions'] = aws_regions
            if review_owner:
                params['ReviewOwner'] = review_owner
            
            response = self.client.update_workload(**params)
            logger.info(f"Updated workload: {workload_id}")
            return response
            
        except ClientError as e:
            logger.error(f"Error updating workload: {str(e)}")
            raise
    
    def delete_workload(self, workload_id: str) -> None:
        """Delete a workload."""
        try:
            self.client.delete_workload(
                WorkloadId=workload_id,
                ClientRequestToken=str(datetime.now().timestamp())
            )
            logger.info(f"Deleted workload: {workload_id}")
        except ClientError as e:
            logger.error(f"Error deleting workload: {str(e)}")
            raise

