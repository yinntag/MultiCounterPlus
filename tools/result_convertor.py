import json
import time

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
periodicity_threshold = 0.3
results = json.load(open('./results/res0_instcount_r50_test.json','r'))
filtered_results = []
for query in results:
    periodicity_converted = []
    periodicity_buffer = []
    for index in range(0, len(query['periodicity_scores'])):
        if query['periodicity_scores'][index] >= periodicity_threshold and periodicity_buffer == []:
            periodicity_buffer.append(index)
        if query['periodicity_scores'][index] < periodicity_threshold and periodicity_buffer != []:
            sum = 0
            for i in range(periodicity_buffer[0],index):     
                sum +=query['periodicity_scores'][i]
            avg_score = sum/(index-periodicity_buffer[0])
            periodicity_converted.extend([[periodicity_buffer[0] , index - 1, avg_score]])
            periodicity_buffer = []
        if (index == len(query['periodicity_scores']) - 1) and periodicity_buffer != []: # If it is an end frame
            sum = 0
            for i in range(periodicity_buffer[0], index+1):
                sum += query['periodicity_scores'][i]
            avg_score = sum / (index - periodicity_buffer[0]+1)
            periodicity_converted.extend([[periodicity_buffer[0], index, avg_score]])
            periodicity_buffer = []
    query.update({'periodicity_converted': periodicity_converted})
    filtered_results.append(query)
json.dump(filtered_results, open('./results/res0_periodicity_converted.json', 'w'), indent=2)
print('Done')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
