from tqdm import tqdm
import requests
import ftplib
import os


# based on gist: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download_file(url, folder=None, n_chunk=1000):
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
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    fname = url.split('/')[-1]
    if folder : 
        fname = os.path.join(folder, fname)
    block_size = 1024
    with open(fname, 'wb') as f, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024) as bar:
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

def ftp_fetch_folder(ftp, folder):
    """Change ftp client working directory and return its content files in a list.

    Parameters
    ----------
    ftp : FTP instance
        ftp client
    folder : str
        folder in ftp server to fetch

    Returns
    -------
    list of str
        files names in ftp folder
    """
    ftp.cwd(folder)
    print(f"Changed directory to remote {folder}")
    return ftp.nlst()

def download_ftp_folder(ftp, folder, store_path, only_files=None):
    """Download all files in ftp folder to local store_path.

    Parameters
    ----------
    ftp : FTP instance
        ftp client
    only_files : list | None
        specific files to download from remote folder, default None download all files.
   folder : str
        folder in ftp server to fetch
    store_path : str
        
    """
    if only_files:
        if isinstance(only_files, list):
            files = [f"{folder}/{f}" for f in only_files]
        else:
            files = [f"{folder}/{only_files}"]
    else:
        files = ftp_fetch_folder(ftp, folder)

    for f in files:
        fpath = os.path.join(store_path, f)
        print(f"Storing file : {f} in {store_path}")
        ftp.retrbinary("RETR " + f, open(fpath, 'wb').write)

    