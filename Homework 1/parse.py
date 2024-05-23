import csv

def parse(filename: str) -> list[dict]:
  '''
  takes a filename and returns attribute information and all the data in array of dictionaries
  '''
  # initialize variables

  out = []  
  # note: you may need to add encoding="utf-8" as a parameter
  with open(filename,'r') as csvfile:
    fileToRead = csv.reader(csvfile)

    headers = next(fileToRead)

    # iterate through rows of actual data
    for row in fileToRead:
      out.append(dict(zip(headers, row)))

  return out