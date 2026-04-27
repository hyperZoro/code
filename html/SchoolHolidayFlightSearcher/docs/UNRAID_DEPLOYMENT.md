# Unraid Deployment Guide

This guide assumes you already have an Nginx Docker container serving other apps, and you do not want those apps to break.

The safest setup is:

- keep your existing Nginx container
- run this app as a second container
- add one new Nginx location: `/flight-searcher/`

That means your current apps keep their current paths, and this app gets its own path.

## 1. Copy The Project To Unraid

Put the whole project folder on your Unraid server, not only `public/`.

Example destination:

```text
/mnt/user/appdata/flight-searcher
```

The folder should contain files like:

```text
Dockerfile
docker-compose.yml
package.json
server.js
public/index.html
src/domain.js
```

The app also creates this folder for stored term dates:

```text
data/
```

On Unraid that means:

```text
/mnt/user/appdata/flight-searcher/data
```

## 2. Build And Start The App Container

Open an Unraid terminal, then go to the project folder:

```bash
cd /mnt/user/appdata/flight-searcher
```

If you want the Gemini fallback for messy term pages, create a `.env` file in this same folder:

```bash
nano .env
```

Add:

```text
FLIGHT_PROVIDER=serpapi
SERPAPI_API_KEY=your-serpapi-key-here
TERM_CACHE_REFRESH_DAYS=183
SERPAPI_CURRENCY=GBP
SERPAPI_GOOGLE_COUNTRY=uk
SERPAPI_LANGUAGE=en
SERPAPI_MAX_DATE_PAIRS=4
SERPAPI_RESULTS_PER_PAIR=5
SERPAPI_RETURN_DETAILS_LIMIT=3
SERPAPI_RETURN_REVERSE_FALLBACK=true
SERPAPI_STOPS=2
SERPAPI_DEEP_SEARCH=false

GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash-lite
```

Save the file. This keeps keys out of the source code and Docker image.

For the first SerpApi test, keep `SERPAPI_MAX_DATE_PAIRS=4` and `SERPAPI_RETURN_DETAILS_LIMIT=3` so the app does not burn through credits too quickly.

Start it:

```bash
docker compose up --build -d
```

Check that it is running:

```bash
docker ps
```

You should see a container named:

```text
flight-searcher
```

## 3. Test The App Before Touching Nginx

Open this in your browser:

```text
http://YOUR_UNRAID_SERVER_IP:3010/
```

Example:

```text
http://192.168.1.50:3010/
```

If that page loads, the app container is working.

Also test the health endpoint:

```text
http://YOUR_UNRAID_SERVER_IP:3010/api/health
```

You should see something like:

```json
{"ok":true,"provider":"mock","service":"school-holiday-flight-searcher"}
```

Do not change Nginx until this direct test works.

## 4. Add A New Nginx Location

Open your existing Nginx config.

Find the relevant `server { ... }` block where your current apps are configured.

Add this new block inside that `server { ... }` block:

```nginx
location = /flight-searcher {
    return 301 /flight-searcher/;
}

location /flight-searcher/ {
    proxy_pass http://YOUR_UNRAID_SERVER_IP:3010/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

Replace:

```text
YOUR_UNRAID_SERVER_IP
```

with your actual server IP.

Example:

```nginx
location = /flight-searcher {
    return 301 /flight-searcher/;
}

location /flight-searcher/ {
    proxy_pass http://192.168.1.50:3010/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

This is intentionally scoped to `/flight-searcher/`, so it should not affect existing apps.

## 5. Reload Or Restart Nginx

Use whatever method you normally use for your Nginx Docker container.

If you use the terminal, it may look like:

```bash
docker restart YOUR_NGINX_CONTAINER_NAME
```

Example:

```bash
docker restart nginx
```

## 6. Open The Proxied URL

Open:

```text
http://YOUR_UNRAID_SERVER_IP/flight-searcher/
```

Example:

```text
http://192.168.1.50/flight-searcher/
```

The status pill should say:

```text
mock provider ready
```

Run a search. If results appear, Nginx and the app container are talking correctly.

## Freemen's Term Dates

The app has a Freemen's helper in the Children section. It loads full-term holiday periods and half-term breaks.

Use it like this:

1. Click `Load Freemen's dates`.
2. Choose a holiday window, such as a half term, Christmas, Easter, or summer holiday.
3. Click `Apply to children`.
4. Check each child card.

The app treats children aged `12` and older as Senior School. Younger children use Junior School dates.

Term dates are stored server-side in:

```text
data/freemens-term-dates.json
```

Click `Load Freemen's dates` to use the stored copy when it is still current.

Click `Refresh stored dates` to fetch the school page again and replace the stored copy.

The app also auto-refreshes when the furthest stored holiday period ends within the next `TERM_CACHE_REFRESH_DAYS`. The default is roughly half a year:

```text
183
```

This date loading is not normally done by AI. The app reads the stored JSON file first, then refreshes by deterministically parsing the Freemen's term-date page. Gemini is only used as a fallback if that parser cannot extract enough dates and `GEMINI_API_KEY` is configured.

## 7. Updating An Existing Install

Use this section when you already have `flight-searcher` running on Unraid and want to replace it with a newer version.

The safe update sequence is:

1. Stop the running container.
2. Keep your `.env` file.
3. Replace the app files.
4. Rebuild the Docker image.
5. Start the container again.
6. Test the health endpoint.

Open an Unraid terminal and go to the app folder:

```bash
cd /mnt/user/appdata/flight-searcher
```

Stop the running container:

```bash
docker compose down
```

If that does not stop it, use:

```bash
docker stop flight-searcher
docker rm flight-searcher
```

Before replacing files, make a backup copy of your `.env` file because it contains your private API keys:

```bash
cp .env .env.backup
```

Now replace the project files in:

```text
/mnt/user/appdata/flight-searcher
```

Important:

- keep `.env`
- keep `.env.backup`
- keep `data/`
- replace app files such as `server.js`, `package.json`, `Dockerfile`, `docker-compose.yml`, `src/`, `public/`, `docs/`, and `test/`
- do not put your API keys into source files

After copying the new files, confirm `.env` is still present:

```bash
ls -la .env
```

If `.env` was accidentally overwritten or removed, restore it:

```bash
cp .env.backup .env
```

If `data/` already exists, leave it in place. It contains cached school term dates:

```text
data/freemens-term-dates.json
```

Rebuild and start:

```bash
docker compose up --build -d
```

Check that the container is running:

```bash
docker ps
```

Check logs:

```bash
docker logs flight-searcher
```

Test the health endpoint:

```text
http://YOUR_UNRAID_SERVER_IP:3010/api/health
```

You should see JSON with:

```json
{"ok":true}
```

If `FLIGHT_PROVIDER=serpapi` is configured, you should also see:

```json
"activeProvider":"serpapi"
```

Finally, open the app:

```text
http://YOUR_UNRAID_SERVER_IP:3010/
```

or, if you configured Nginx:

```text
http://YOUR_UNRAID_SERVER_IP/flight-searcher/
```

Hard refresh the browser after an update so old JavaScript or CSS is not reused:

```text
Ctrl + F5
```

On macOS:

```text
Cmd + Shift + R
```

## 8. If Something Goes Wrong

First check the app directly:

```text
http://YOUR_UNRAID_SERVER_IP:3010/
```

If direct access works but `/flight-searcher/` does not, the issue is in the Nginx location.

Check app logs:

```bash
docker logs flight-searcher
```

Restart the app:

```bash
docker restart flight-searcher
```

Stop the app:

```bash
docker compose down
```

## 9. Why Port 3010?

Inside the container, the app listens on port `3000`.

On Unraid, `docker-compose.yml` maps it to host port `3010`:

```yaml
ports:
  - "3010:3000"
```

So the app is reachable on:

```text
http://YOUR_UNRAID_SERVER_IP:3010/
```

You can change `3010` if that port is already used.

## 10. Future Functionality

This deployment style does not block future work.

Later, the backend can safely hold:

- real flight provider API keys
- saved family/school defaults
- manual flight import
- school page parsing
- AI-assisted date extraction

Those should stay on the server side, not in browser-only static files.
