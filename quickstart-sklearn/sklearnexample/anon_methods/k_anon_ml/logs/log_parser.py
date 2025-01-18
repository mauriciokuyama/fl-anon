import csv

method = 'cb'

with open(f'./{method}.txt', "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(f'adult_anon_metrics_ncp_{method}.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['k', 'ncp'])

    for line in lines:
        if line.startswith('K='):
            k = int(line.split('=')[1])
        elif line.startswith('NCP'):
            ncp = float(line.split()[1][:-1])
            csvwriter.writerow([k, ncp])
