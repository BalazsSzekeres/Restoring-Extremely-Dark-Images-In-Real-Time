import sys
import typing

import google.cloud.compute_v1 as compute_v1

def stop_instance(project_id, zone, machine_name):
    instance_client = compute_v1.InstancesClient()
    operation_client = compute_v1.ZoneOperationsClient()

    print(f"Stopping {machine_name} from {zone}...")
    operation = instance_client.stop(
        project=project_id, zone=zone, instance=machine_name
    )
    while operation.status != compute_v1.Operation.Status.DONE:
        operation = operation_client.wait(
            operation=operation.name, zone=zone, project=project_id
        )
    if operation.error:
        print("Error during stop:", operation.error, file=sys.stderr)
    if operation.warnings:
        print("Warning during stop:", operation.warnings, file=sys.stderr)
    print(f"Instance {machine_name} stopped.")
    return

stop_instance('project-ID', 'zone', 'machine-name')'