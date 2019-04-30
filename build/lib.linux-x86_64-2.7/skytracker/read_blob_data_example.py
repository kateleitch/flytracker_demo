import sys
from blob_data_tools import load_blob_data

blob_data_filename = sys.argv[1]
blob_data = load_blob_data(blob_data_filename)


for item in blob_data:
    print 'frame:      {0}'.format(item['frame'])
    print '# blobs:    {0}'.format(len(item['blobs']))
    print 'blobs list: {0}'.format(item['blobs'])
    print 
