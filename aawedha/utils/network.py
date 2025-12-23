from io import BytesIO
from tqdm import tqdm
import requests
import ftplib
import pycurl
import glob
import os


# based on gist: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download_file(url, folder=None, timeout=((3, 5)), n_chunk=1000):
    """Download file from url and stored in folder.

    Parameters
    ----------
    url : str
        file link
    folder : str, optional
        folder path where to save file, by default None saves file in working directory.
    n_chunk : int, optional
        streaming chunks count, by default 1000
    """
    if not isinstance(url, list):
        url = [url]
        
    block_size = 8192
    with requests.Session() as session:
        # resp = requests.get(url, stream=True, timeout=(3,5))
        for u in url:
            resp = session.get(u, stream=True, timeout=timeout)
            total = int(resp.headers.get('content-length', 0))
            # check if the file is already downloaded            
            fname = u.split('/')[-1]
            if folder : 
                fname = os.path.join(folder, fname)   
            if os.path.exists(fname):
                print(f"File {fname} already exists, skipping download.")
                continue     
            print(f"Downloading {u} to {folder} with size {total} bytes")
            with open(fname, 'wb') as f, tqdm(desc=fname, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
                for data in resp.iter_content(chunk_size=n_chunk * block_size):
                    size = f.write(data)
                    bar.update(size)

def connect_ftp(url, user='anonymous', password=''):
    """Connect and login a ftp client to url host for ftp files download.

    Parameters
    ----------
    url : str
        host url
    user : str, optional
        user name, by default 'anonymous'
    password : str, optional
        user's password, by default ''

    Returns
    -------
    FTP
        instance of a FTP class
    """
    ftp = ftplib.FTP(url)
    print(f"Connection successful to {url}")
    ftp.login(user, password)
    return ftp

def ftp_fetch_folder(ftp, folder, pattern=None):
    """Change ftp client working directory and return its content files in a list.

    Parameters
    ----------
    ftp : FTP instance
        ftp client
    folder : str
        folder in ftp server to fetch
    pattern : str
        all files by extension or name to fetch and download from remote folder, default None download all 
        files.    
    Returns
    -------
    list of str
        files names in ftp folder
    """
    ftp.cwd(folder)
    print(f"Changed directory to remote {folder}")
    if pattern:
        return ftp.nlst(pattern)
    else:
        return ftp.nlst()


def download_ftp_folder(ftp, folder, store_path, only_files=None, pattern=None):
    """Download all files in ftp folder to local store_path.

    Parameters
    ----------
    ftp : FTP instance
        ftp client    
   folder : str
        folder in ftp server to fetch
    store_path : str
        folder path where to store files locally.
    only_files : list | None
        specific files by name to download from remote folder, default None download all files. 
    pattern : str
        all files by extension or name to fetch and download from remote folder, default None download all 
        files.    
    """
    inside_wd = False
    if only_files:
        if isinstance(only_files, list):
            files = [f"{folder}/{f}" for f in only_files]
        else:
            files = [f"{folder}/{only_files}"]
    else:
        files = ftp_fetch_folder(ftp, folder, pattern)
        inside_wd = True
    
    if not inside_wd:
      ftp.cwd(folder)
      
    for f in files:
        fname = f
        if not inside_wd:
          fname = f.split('/')[-1]
        fpath = os.path.join(store_path, fname)
        print(f"Storing file : {f} in {store_path}")
        ftp.retrbinary("RETR " + fname, open(fpath, 'wb').write)
        # ftp.retrbinary("RETR " + fname, open(fpath, 'wb').write)
        # ftp.retrbinary("RETR " + f, open(f, 'wb').write)
        # ftp.retrbinary("RETR " + f, open(fpath, 'wb').write)

def download_pycurl(url, output_filename):
    """Downloads a file from a given URL using pycurl and saves it to a specified filename."
    Parameters
    ----------
    url : str
        The URL of the file to download.
    output_filename : str
        The name of the file to save the downloaded content to.
    Returns
    -------
    None
    Raises
    ------
    pycurl.error
        If a pycurl-specific error occurs during the download.
    Exception
        For any other unexpected errors.
    Notes
    -----
    The function prints status messages to indicate the progress and result of the download.
    If the HTTP status code is not 200, a warning is printed.
    """
    buffer = BytesIO() # Create a BytesIO object to store the data in memory initially
    c = pycurl.Curl()

    try:
        # Set the URL to download
        c.setopt(c.URL, url)

        # Set the callback function for writing data
        # pycurl will call this function with chunks of data
        # We write these chunks directly to the output file
        with open(output_filename, 'wb') as f:
            c.setopt(c.WRITEDATA, f)

            # Perform the request
            print(f"Starting download from: {url}")
            c.perform()
            print(f"Download complete! File saved to: {output_filename}")

        # Get response code for success check
        status_code = c.getinfo(pycurl.HTTP_CODE)
        if status_code != 200:
            print(f"Warning: HTTP Status Code: {status_code}")

    except pycurl.error as e:
        error_code, error_message = e.args
        print(f"PycURL Error ({error_code}): {error_message}")
        print(f"Could not download {url}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Always close the curl handle
        c.close()


def remote_dataset_size(url):
    """Get the size of a remote dataset file.

    Parameters
    ----------
    path : str
        local folder path where the file is stored.
    url : str
        remote file url.

    Returns
    -------
    int
        size of the remote file in bytes.
    """
    with requests.Session() as session:
        resp = session.get(url, stream=True)
        if resp.status_code == 200:
            return int(resp.headers.get('content-length', 0))
        else:
            print(f"Failed to retrieve size for {url}, status code: {resp.status_code}")
            return 0


def check_size(ftp, path, remote_folder):
    """Compare size of local file with remote one, used to re-download a file
    in case of uncomplete download. 

    Parameters
    ----------
    ftp : FTP instance
        ftp client 
    path : str
        local folder path
    remote_folder : str
        remote folder path

    Returns
    -------
    bool
        True if all files in folder has been downloaded correctly, False otherwise.
    """
    files = glob.glob(f"{path}/*")
    state = []
    for f in files:
        fname = f.split('/')[-1]
        local_size = os.path.getsize(f)
        remote_size =  ftp.size(f"{remote_folder}/{fname}")
        if local_size == remote_size:
            state.append(1)
        else:
            state.append(0)
    return all(state)     

    