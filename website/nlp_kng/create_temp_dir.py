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

def createSvoTempDir():
    tmp_dir_name = get_temp_dir()
    print('tmp_dir_name -->',tmp_dir_name)
    svo_df_temp_path = tmp_dir_name +'/svo_df_temp' #os.path.join(tmp_dir_name, '/svo_df_temp')
    print(f'svo_df_temp_path --> {svo_df_temp_path}')
    isExist = os.path.exists(svo_df_temp_path)
    try:
        os.mkdir(svo_df_temp_path)
    except Exception as e:
        print('Svo_dir already exists', e)
    print(svo_df_temp_path)
    return svo_df_temp_path

def get_temp_dir():
    return directory_name
