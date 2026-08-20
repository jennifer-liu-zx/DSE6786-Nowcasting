# GDP Nowcasting
## Set-up
### Setting up Virtual Environment (Optional but Recommended)
1. **Create the Venv**:
Copy the code into your terminal
   1. _Windows_: `python -m venv .venv`
   2. _Mac/Linux_: `python3 -m venv .venv`
1. **Activate Virtual Env**
   1. _Windows_: `.venv\Scripts\activate`
   2. _Mac/Linux_: `source .venv/bin/activate`
   
### Installing dependencies
1. Run the following command in your terminal: 
`pip install -r requirements.txt`

### Setting up Environment
1. Ensure you have a `.env` file in your root directory. If it does not exist, create it.
2. Ensure the following lines are present.
```
SUPABASE_URL = "https://uhxliyubyjwtpxjcfedk.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_tGsiADOfozJKrf5npbdyew_lS4y4otL"
```
3. If you are a developer, the SUPABASE_ANON_KEY should be replaced by your private Anon Key. You should also have the `SUPABASE_SERVICE_KEY`.

### Running The Application
1. Run the following command in your terminal to launch the application
`shiny run app.py`
2. The local host link will be shown in the terminal.
>> Local Host Link: `http://127.0.0.1:8000`
4. Copy the link and paste it into the Address bar of your preferred browser.
5. To quit the application, go to your terminal and press `CTRL + C`
