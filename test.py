import csv


csvfile_for_read = open('sampletmbz.csv','r')
count =0
reader = csv.reader(csvfile_for_read)
for i in reader:
    print(i[0])
    count = count +1

with open('sampletmbz.csv','a') as csvfile_for_write:
    newWriter = csv.writer(csvfile_for_write)
    newWriter.writerow([3,'123'])






csvfile_for_write.close()
csvfile_for_read.close()


print(count)