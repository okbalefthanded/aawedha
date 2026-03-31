from pathlib import Path
from io import BytesIO
from tqdm import tqdm
import requests
import ftplib
import pycurl
import glob
import os

def download_file(
    url: str | list[str],
    folder: str | None = None,
    timeout: tuple[int, int] = (3, 5),
    n_chunk: int = 1000,
    overwrite: bool = False,
    resume: bool = True,
) -> list[Path]:
    """Download file(s) from url and store in folder.

    Parameters
    ----------
    url : str | list[str]
        File link or list of file links.
    folder : str, optional
        Folder path where to save file(s). Defaults to current working directory.
    timeout : tuple[int, int], optional
        (connect, read) timeout in seconds. Default is (3, 5).
    n_chunk : int, optional
        Streaming chunk count multiplier (chunk_size = n_chunk * 8192). Default is 1000.
    overwrite : bool, optional
        If True, re-download even if file exists. Default is False.
    resume : bool, optional
        If True, resume partially downloaded files. Default is True.

    Returns
    -------
    list[Path]
        List of paths to downloaded files.

    Raises
    ------
    requests.HTTPError
        If the server returns an error status code.
    OSError
        If the folder cannot be created or the file cannot be written.
    """
    urls = [url] if isinstance(url, str) else url
    block_size = 8192
    chunk_size = n_chunk * block_size
    downloaded_files: list[Path] = []

    # Create folder if it doesn't exist
    output_dir = Path(folder) if folder else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        for u in urls:
            fname = output_dir / Path(u.split("?")[0].split("/")[-1])  # strip query params from filename

            # --- Skip if already fully downloaded ---
            if fname.exists() and not overwrite:
                existing_size = fname.stat().st_size
                try:
                    head = session.head(u, timeout=timeout, allow_redirects=True)
                    head.raise_for_status()
                    remote_size = int(head.headers.get("content-length", -1))
                    if remote_size < 0 or existing_size == remote_size:
                        print(f" Already downloaded: {fname}")
                        downloaded_files.append(fname)
                        continue
                except requests.RequestException:
                    # Can't verify remote size — skip conservatively
                    print(f" Already exists (could not verify size): {fname}")
                    downloaded_files.append(fname)
                    continue

            # --- Resumable download support ---
            resume_header = {}
            initial_pos = 0
            write_mode = "wb"

            if resume and fname.exists() and not overwrite:
                initial_pos = fname.stat().st_size
                resume_header = {"Range": f"bytes={initial_pos}-"}
                write_mode = "ab"
                print(f" Resuming {fname.name} from byte {initial_pos:,}")

            # --- Fetch ---
            try:
                resp = session.get(u, stream=True, timeout=timeout, headers=resume_header)
                resp.raise_for_status()
            except requests.HTTPError as e:
                print(f" HTTP error for {u}: {e}")
                continue
            except requests.ConnectionError as e:
                print(f" Connection error for {u}: {e}")
                continue
            except requests.Timeout:
                print(f" Timed out connecting to {u}")
                continue

            total = int(resp.headers.get("content-length", 0)) + initial_pos

            print(f"⬇ Downloading {u}\n  → {fname}  ({total / 1024**2:.1f} MB)")

            try:
                with (
                    open(fname, write_mode) as f,
                    tqdm(
                        desc=fname.name,
                        total=total,
                        initial=initial_pos,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                        dynamic_ncols=True,
                    ) as bar,
                ):
                    for data in resp.iter_content(chunk_size=chunk_size):
                        size = f.write(data)
                        bar.update(size)
            except OSError as e:
                print(f"✘ Failed to write {fname}: {e}")
                continue

            downloaded_files.append(fname)
            print(f" Saved: {fname}")

    return downloaded_files

# based on gist: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download_file_legacy(url, folder=None, timeout=((3, 5)), n_chunk=1000):
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


def remote_dataset_size(urls):
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
    size = 0
    with requests.Session() as session:
        for url in urls:
            resp = session.get(url, stream=True)
            if resp.status_code == 200:
                size += int(resp.headers.get('content-length', 0))
            else:
                print(f"Failed to retrieve size for {url}, status code: {resp.status_code}")       
    return size

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

    