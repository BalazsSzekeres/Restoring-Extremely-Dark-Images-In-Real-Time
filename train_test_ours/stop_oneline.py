import json
import googleapiclient.discovery


compute = googleapiclient.discovery.build('compute', 'v1')
compute.instances().stop(project='mygcpproject-123', zone='northamerica-northeast2-a', instance='instance_name').execute()