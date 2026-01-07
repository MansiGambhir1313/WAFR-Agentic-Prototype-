"""
List existing AWS Well-Architected Tool workloads
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wafr.agents.wa_tool_client import WellArchitectedToolClient


def main() -> None:
    print("=" * 70)
    print("  AWS WELL-ARCHITECTED TOOL - LIST WORKLOADS")
    print("=" * 70 + "\n")
    
    client = WellArchitectedToolClient()
    workloads = client.list_workloads()
    
    if not workloads:
        print("No workloads found.")
        return
    
    print(f"Found {len(workloads)} workload(s):\n")
    
    for i, workload in enumerate(workloads, 1):
        workload_id = workload.get('WorkloadId', 'N/A')
        workload_name = workload.get('WorkloadName', 'N/A')
        environment = workload.get('Environment', 'N/A')
        updated_at = workload.get('UpdatedAt', 'N/A')
        
        print(f"{i}. {workload_name}")
        print(f"   ID: {workload_id}")
        print(f"   Environment: {environment}")
        print(f"   Updated: {updated_at}")
        print()

if __name__ == '__main__':
    main()

