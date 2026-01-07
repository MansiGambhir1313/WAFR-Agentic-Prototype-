"""
List all AWS Official lenses available via API
"""
import json
from typing import List, Dict, Any

import boto3


def list_all_official_lenses(region_name: str = 'us-east-1') -> List[Dict[str, Any]]:
    """
    Get all AWS official lenses including Gen AI.
    
    Args:
        region_name: AWS region name (default: us-east-1)
        
    Returns:
        List of lens dictionaries from AWS API
    """
    client = boto3.client('wellarchitected', region_name=region_name)
    
    all_lenses = []
    next_token = None
    
    while True:
        params = {
            'LensType': 'AWS_OFFICIAL',
            'LensStatus': 'PUBLISHED',
            'MaxResults': 50
        }
        
        if next_token:
            params['NextToken'] = next_token
        
        try:
            response = client.list_lenses(**params)
            lenses = response.get('LensSummaries', [])
            all_lenses.extend(lenses)
            
            next_token = response.get('NextToken')
            if not next_token:
                break
        except Exception as e:
            print(f"Error listing lenses: {e}")
            break
    
    return all_lenses

def main() -> None:
    """Main entry point for listing AWS lenses."""
    print("=" * 70)
    print("AWS OFFICIAL LENSES (via API)")
    print("=" * 70)
    
    lenses = list_all_official_lenses()
    
    print(f"\nFound {len(lenses)} AWS Official lenses:\n")
    
    for lens in lenses:
        alias = lens.get('LensAlias', 'N/A')
        name = lens.get('LensName', 'N/A')
        lens_type = lens.get('LensType', 'N/A')
        status = lens.get('LensStatus', 'N/A')
        
        print(f"  • {alias:30} | {name}")
        print(f"    Type: {lens_type}, Status: {status}")
        print()
    
    # Save to JSON for reference
    with open('all_aws_lenses.json', 'w', encoding='utf-8') as f:
        json.dump(lenses, f, indent=2, default=str)
    
    print("\n✓ Saved to all_aws_lenses.json")
    print("\nLens Aliases (for use in code):")
    aliases = [l.get('LensAlias') for l in lenses if l.get('LensAlias')]
    print(f"  {', '.join(aliases)}")


if __name__ == '__main__':
    main()

