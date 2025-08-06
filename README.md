# Langfuse Latency Monitor

Langfuse Latency Monitor is a Python tool for simulating, monitoring, and analyzing the latency and error rates of OpenAI chat completions, with detailed tracking and visualization in the Langfuse dashboard. It supports error and slow-call simulation, session-level statistics, and is ideal for stress-testing LLM-based chat systems.

## Features
- **Latency & Error Monitoring:** Tracks each chat turn's latency and errors, logs to Langfuse.
- **Simulation:** Randomly simulates slow network calls and API errors for robust testing.
- **Session Statistics:** Aggregates and reports per-session and overall performance.
- **Langfuse Integration:** All metrics and errors are sent to Langfuse for visualization and alerting.

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/nishanttomar21/langfuse-latency-monitor.git
   cd langfuse-latency-monitor
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment:**
   - Create a `.env` file with your OpenAI and Langfuse credentials:
     ```env
     OPENAI_API_KEY=your-openai-key
     LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
     LANGFUSE_SECRET_KEY=your-langfuse-secret-key
     LANGFUSE_HOST=https://cloud.langfuse.com
     ```

## Usage
Run the main script to simulate multiple monitored chat sessions:
```bash
python main.py
```
- The script will print session and turn-level stats, simulate errors/latency, and send all data to Langfuse.
- Check your Langfuse dashboard for detailed metrics, slow call alerts, and error tracking.

## Flow Overview

```mermaid
flowchart TD
    A["User/Script Starts Session"] --> B["Simulate Conversation Session"]
    B --> C["For Each Turn:"]
    C --> D["Simulate Network Conditions (optional)"]
    C --> E["Simulate API Errors (optional)"]
    C --> F["Call OpenAI API (chat completion)"]
    F --> G["Measure Latency"]
    F --> H["Capture Errors"]
    G --> I["Log to Langfuse (latency, performance)"]
    H --> J["Log to Langfuse (error details)"]
    I --> K["Update Session Stats"]
    J --> K
    K --> L["Next Turn or End"]
    L --> M["Session Summary & Report"]
    M --> N["Flush Data to Langfuse"]
    N --> O["View Metrics & Alerts in Langfuse Dashboard"]
```

## Screenshots
Below are some screenshots of the tool and Langfuse dashboard:

| Screenshot | Description |
|------------|-------------|
| ![Screenshot 1](assets/screenshot%201.png) | Example session output in terminal |
| ![Screenshot 2](assets/screenshot%202.png) | Langfuse dashboard: latency metrics |
| ![Screenshot 3](assets/screenshot%203.png) | Langfuse dashboard: error tracking |
| ![Screenshot 4](assets/screenshot%204.png) | Langfuse dashboard: session details |
| ![Screenshot 5](assets/screenshot%205.png) | Langfuse dashboard: performance alerts |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
