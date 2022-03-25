import json
import googleapiclient.discovery


compute = googleapiclient.discovery.build('compute', 'v1')

result = compute.instances().list(project='mygcpproject-123', zone='northamerica-northeast2-a').execute()
#print(json.dumps(result, indent=2))


#print(len(result)) #returns: 4 (id, items, selfLink, kind)
vms_list=result['items']
num_vms=len(vms_list)

# loop through your instances list and find the ones with 'save_money' in the output (which is your label's key name -- it doesn't matter what the value is)
for i in range(num_vms):
    if "save_money" in json.dumps(vms_list[i]):
        print("stopping {}".format(vms_list[i]['name']))
        compute.instances().stop(project='mygcpproject-123', zone='northamerica-northeast2-a', instance=vms_list[i]['name']).execute()