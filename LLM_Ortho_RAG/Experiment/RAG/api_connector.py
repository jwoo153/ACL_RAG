import requests
import pandas as pd
import pathlib
import time


directory_path = pathlib.Path(__file__).parent.absolute()


class api_connector: 


    def __init__(self, 
                 api_key: str,
                 api_url: str = 'http://localhost:3001/api'):
        """Initializes an api connector object.
        
        Parameters
        ----------
        - api_key : str
            The API key to authenticate with the AnythingLLM API.
        - api_url : str
            The URL of the AnythingLLM API.
        """
        self.api_key_ = api_key
        self.api_url_ = api_url
        self.headers_ = {
            'Authorization': f'Bearer {self.api_key_}', 
        }
        self.workspaces_metadata_ = pd.DataFrame(self._workspaces())
        self.workspace_name_ = None
        self.worksapce_slug_ = None


    def _workspaces(self):
        """Returns all workspaces recognized
        
        Returns
        -------
        - list[dict]. Each dict contains metadata for a workspace.
        """
        endpoint_url = f'{self.api_url_}/v1/workspaces'
        response = requests.get(endpoint_url, headers=self.headers_)
        response.raise_for_status()
        return response.json()['workspaces']
        

    def workspaces_metadata(self) -> pd.DataFrame:
        """Retrieves metadata for all workspaces.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing metadata for all workspaces.
        """
        return self.workspaces_metadata_
    
    def set_workspace(self, name: str):
        """Sets the workspace and slug based on the 
        provided workspace name. Workspaces metadata may be 
        retrieved using the workspaces_metadata() method."""
        workspace_df = self.workspaces_metadata_[
            self.workspaces_metadata_['name'] == name]
        if len(workspace_df) >= 2:
            raise ValueError(f'Workspace name {name} is not unique.')
        elif len(workspace_df) == 0:
            raise ValueError(f'Workspace name {name} not found.')
        else:
            self.workspace_name_ = name
            self.workspace_slug_ = workspace_df['slug'].item()
        
    def chat(self, prompt: str, recursion_depth: int = 0) -> str:
        """Generates a response.
        
        Parameters
        ----------
        - prompt : str.
            The prompt to generate a response for.
        
        Returns
        -------
        - str.
            The response generated.
        """
        self._verify_setup()
        endpoint_url = f'{self.api_url_}/v1/workspace/' +\
            f'{self.workspace_slug_}/chat'
        response = requests.post(
            endpoint_url, headers=self.headers_, json={'message': prompt, 
                                                       'mode': 'chat'})
        if response.status_code == 500 and recursion_depth < 2:
            print(response.json())
            print('sleeping for 60 seconds...')
            time.sleep(60)
            self.chat(prompt, recursion_depth + 1)
    
        response.raise_for_status()
        return response.json()['textResponse']


    def query(self, prompt: str, recursion_depth: int = 0) -> str:
        """Generates a response.
        
        Parameters
        ----------
        - prompt : str.
            The prompt to generate a response for.
        
        Returns
        -------
        - str.
            The response generated .
        """
        self._verify_setup()
        endpoint_url = f'{self.api_url_}/v1/workspace/' +\
            f'{self.workspace_slug_}/chat'
        response = requests.post(
            endpoint_url, headers=self.headers_, json={'message': prompt, 
                                                       'mode': 'query'})
        if response.status_code == 500 and recursion_depth < 2:
            print(response.json())
            print('sleeping for 60 seconds...')
            time.sleep(60)
            self.query(prompt, recursion_depth + 1)
    
        response.raise_for_status()
        return response.json()['textResponse']
    

    def _authenticate(self):
        """Authenticates the API key with the AnythingLLM API.
        Error will be raised if method fails."""
        endpoint_url = f'{self.api_url_}/v1/auth'
        response = requests.get(endpoint_url, headers=self.headers_)
        response.raise_for_status()


    def _verify_setup(self):
        """Raises Runtime Error if workspace is not set."""
        if self.workspace_name_ is None or self.workspace_slug_ is None:
            raise RuntimeError('Workspace not set. Use set_workspace() method.')



