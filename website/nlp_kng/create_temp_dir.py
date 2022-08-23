import os
import tempfile
directory_name = ''
def createTempDir():
    current_path = os.path.abspath(os.path.dirname(__file__))
    temp_path = os.path.join(current_path, '../../tmp')
    tempfile.tempdir = temp_path
    global directory_name 
    directory_name = tempfile.mkdtemp()
    print(directory_name)
    return directory_name

def get_temp_dir():
    return directory_name

# if __name__ == '__main__':
#     tmpdir = create_temp_dir()
#     print(tmpdir)
