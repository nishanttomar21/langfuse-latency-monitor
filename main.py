import os
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import get_client
from datetime import datetime
import time
import random

# Load environment variables
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
langfuse = get_client()

# Performance thresholds for monitoring
LATENCY_THRESHOLD_MS = 3000  # 3 seconds
SLOW_CALL_SIMULATION_CHANCE = 0.3  # 30% chance of slow call
ERROR_SIMULATION_CHANCE = 0.9  # 90% chance of error


def create_session_metadata(session_id, user_id, turn_number, total_turns):
    """Create metadata for session tracking"""
    return {
        "session_id": session_id,
        "user_id": user_id,
        "turn_number": turn_number,
        "total_turns": total_turns,
        "timestamp": datetime.now().isoformat(),
    }


def create_turn_metadata(session_id, user_id, turn_number, model, temperature, max_tokens, latency_ms=None, error_type=None):
    """Create metadata for individual turn tracking with performance data"""
    metadata = {
        "session_id": session_id,
        "user_id": user_id,
        "turn_number": turn_number,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": datetime.now().isoformat(),
    }

    # Add performance metrics
    if latency_ms is not None:
        metadata["latency_ms"] = latency_ms
        metadata["is_slow_call"] = latency_ms > LATENCY_THRESHOLD_MS
        metadata["performance_tier"] = (
            "fast" if latency_ms < 1000 else
            "normal" if latency_ms < LATENCY_THRESHOLD_MS else
            "slow"
        )

    # Add error information
    if error_type:
        metadata["error_type"] = error_type
        metadata["has_error"] = True
    else:
        metadata["has_error"] = False

    return metadata


def simulate_network_conditions():
    """Simulate various network conditions for testing"""
    # Randomly simulate slow network conditions
    if random.random() < SLOW_CALL_SIMULATION_CHANCE:
        delay = random.uniform(2, 5)  # 2-5 second delay
        print(f" Simulating slow network (adding {delay:.1f}s delay)")
        time.sleep(delay)
        return True
    return False


def simulate_api_errors():
    """Simulate API errors for testing"""
    if random.random() < ERROR_SIMULATION_CHANCE:
        error_types = [
            ("rate_limit", "Rate limit exceeded"),
            ("timeout", "Request timeout"),
            ("invalid_request", "Invalid request parameters"),
            ("service_unavailable", "Service temporarily unavailable")
        ]
        error_type, error_msg = random.choice(error_types)
        print(f" Simulating API error: {error_msg}")
        raise Exception(f"Simulated {error_type}: {error_msg}")


def chat_turn_with_monitoring(prompt, conversation_history, session_id, user_id, turn_number, model="gpt-3.5-turbo", temperature=0.7, max_tokens=150, enable_simulation=True):
    """Execute a single chat turn with comprehensive monitoring and error handling"""

    # Start timing the API call
    start_time = time.time()
    latency_ms = None
    error_type = None
    response_text = None

    # Initial metadata (will be updated with performance data)
    initial_metadata = create_turn_metadata(session_id, user_id, turn_number, model, temperature, max_tokens)

    with langfuse.start_as_current_generation(
            name=f"chat_turn_{turn_number}",
            model=model,
            input={"prompt": prompt, "conversation_context": len(conversation_history)},
            metadata=initial_metadata,
    ) as generation:

        try:
            # Simulate network conditions if enabled
            was_slow_simulated = False
            if enable_simulation:
                was_slow_simulated = simulate_network_conditions()
                simulate_api_errors()  # May raise an exception

            # Build messages with conversation history
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Keep responses concise but informative."}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

            # Make the actual OpenAI API call
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Calculate latency
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            response_text = response.choices[0].message.content

            # Create updated metadata with performance metrics
            final_metadata = create_turn_metadata(
                session_id, user_id, turn_number, model, temperature, max_tokens,
                latency_ms=latency_ms
            )

            # Performance logging
            if latency_ms > LATENCY_THRESHOLD_MS:
                print(f"  SLOW CALL DETECTED: {latency_ms}ms (threshold: {LATENCY_THRESHOLD_MS}ms)")
            else:
                print(f" Call completed in {latency_ms}ms")

            if was_slow_simulated:
                final_metadata["simulation_type"] = "slow_network"

            # Update generation with successful response
            if generation is not None:
                generation.update(
                    output=response_text,
                    metadata=final_metadata,
                    usage={
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                )

            return response_text, False

        except Exception as e:
            # Calculate latency even for failed calls
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            # Determine error type
            error_str = str(e).lower()
            if "rate" in error_str and "limit" in error_str:
                error_type = "rate_limit"
            elif "timeout" in error_str:
                error_type = "timeout"
            elif "invalid" in error_str:
                error_type = "invalid_request"
            elif "service" in error_str or "unavailable" in error_str:
                error_type = "service_unavailable"
            else:
                error_type = "unknown_error"

            # Create metadata with error information
            error_metadata = create_turn_metadata(
                session_id, user_id, turn_number, model, temperature, max_tokens,
                latency_ms=latency_ms, error_type=error_type
            )
            error_metadata["error_message"] = str(e)
            error_metadata["simulation_type"] = "api_error" if enable_simulation else "real_error"

            print(f" ERROR: {e} (took {latency_ms}ms)")

            # Update generation with error information
            if generation is not None:
                generation.update(
                    output=f"Error: {str(e)}",
                    metadata=error_metadata,
                    level="ERROR"
                )

            # For simulation purposes, return a fallback response instead of crashing
            if enable_simulation and "Simulated" in str(e):
                return f" Simulated error occurred: {error_type}. This would normally be handled by retry logic.", True  # mark as failure
            else:
                raise e


def simulate_conversation_session_with_monitoring(user_id, session_id, conversation_turns, model="gpt-3.5-turbo", enable_simulation=True):
    """Simulate a complete conversation session with comprehensive monitoring"""
    session_metadata = create_session_metadata(session_id, user_id, 0, len(conversation_turns))

    # Add monitoring metadata to session
    session_metadata["monitoring_enabled"] = True
    session_metadata["latency_threshold_ms"] = LATENCY_THRESHOLD_MS
    session_metadata["simulation_enabled"] = enable_simulation

    with langfuse.start_as_current_span(
            name=f"monitored_session_{session_id}",
            input={"user_id": user_id, "session_id": session_id},
            metadata=session_metadata,
    ) as session_span:

        conversation_history = []
        session_stats = {
            "total_turns": len(conversation_turns),
            "successful_turns": 0,
            "failed_turns": 0,
            "slow_turns": 0,
            "total_latency_ms": 0,
            "errors": []
        }

        print(f"\nStarting MONITORED conversation session: {session_id} (User: {user_id})")
        print(f"Monitoring: Latency threshold {LATENCY_THRESHOLD_MS}ms, Simulation: {enable_simulation}")
        print("-" * 80)

        for turn_num, user_prompt in enumerate(conversation_turns, 1):
            print(f"\nTurn {turn_num}/{len(conversation_turns)}")
            print(f" {user_id}: {user_prompt}")

            try:
                # Execute the monitored chat turn
                bot_response, was_failure = chat_turn_with_monitoring(
                    prompt=user_prompt,
                    conversation_history=conversation_history,
                    session_id=session_id,
                    user_id=user_id,
                    turn_number=turn_num,
                    model=model,
                    temperature=0.7 + turn_num * 0.05,
                    max_tokens=150,
                    enable_simulation=enable_simulation
                )

                print(f" Bot: {bot_response}")

                # Update session statistics
                if was_failure:
                    session_stats["failed_turns"] += 1
                    session_stats["errors"].append(bot_response)
                else:
                    session_stats["successful_turns"] += 1

                # Add to conversation history for context in next turns
                conversation_history.append({"role": "user", "content": user_prompt})
                conversation_history.append({"role": "assistant", "content": bot_response})

            except Exception as e:
                session_stats["failed_turns"] += 1
                session_stats["errors"].append(str(e))
                print(f" Turn {turn_num} failed: {e}")

                # Add user message but not assistant response for failed turns
                conversation_history.append({"role": "user", "content": user_prompt})

            # Brief pause between turns
            time.sleep(0.5)

        # Calculate session-level metrics
        session_stats["success_rate"] = session_stats["successful_turns"] / len(conversation_turns)     # type: ignore
        session_stats["error_rate"] = session_stats["failed_turns"] / len(conversation_turns)           # type: ignore

        # Update session span with final statistics
        if session_span is not None:
            session_span.update(
                output={
                    "session_completed": True,
                    "statistics": session_stats,
                    "final_context_length": len(conversation_history)
                }
            )

        print(f"\n Session {session_id} Statistics:")
        print(f"    Successful turns: {session_stats['successful_turns']}/{len(conversation_turns)}")
        print(f"    Failed turns: {session_stats['failed_turns']}/{len(conversation_turns)}")
        print(f"    Success rate: {session_stats['success_rate']:.1%}")

        return conversation_history, session_stats


def main():
    """Run monitored conversation sessions with performance tracking and error simulation"""
    print()
    print("-" * 80)
    print(" MONITORED Multi-User Conversation Tracking with Langfuse")
    print(" Features: Latency monitoring, Error tracking, Performance alerts")
    print("-" * 80)
    print()

    # Test scenarios with different conditions
    test_scenarios = {
        "nishant_tomar": {
            "session_normal": [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "Can you give me an example?"
            ]
        },
        "megha_singh": {
            "session_stress_test": [
                "Tell me about quantum computing",
                "What are the practical applications?",
                "How does it compare to classical computing?",
                "What are the current limitations?",
                "When will it become mainstream?"  # More turns to increase chance of errors/slow calls
            ]
        },
        "rajat_rajput": {
            "session_quick": [
                "Hello, how are you?",
                "What's the weather like?"
            ]
        }
    }

    all_session_stats = []

    # Process all test scenarios
    for user_id, sessions in test_scenarios.items():
        for session_name, turns in sessions.items():
            full_session_id = f"{user_id}_{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            print(f"\n Processing: {full_session_id}")

            try:
                conversation_history, stats = simulate_conversation_session_with_monitoring(
                    user_id=user_id,
                    session_id=full_session_id,
                    conversation_turns=turns,
                    model="gpt-3.5-turbo",
                    enable_simulation=True  # Enable error/latency simulation
                )

                all_session_stats.append({
                    "user_id": user_id,
                    "session_id": full_session_id,
                    "stats": stats
                })

            except Exception as e:
                print(f" Session {full_session_id} failed completely: {e}")

            # Brief pause between sessions
            time.sleep(1)

    # Generate summary report
    print(f"\n PERFORMANCE SUMMARY REPORT")
    print("=" * 80)

    total_sessions = len(all_session_stats)
    total_successful_turns = sum(s["stats"]["successful_turns"] for s in all_session_stats)
    total_failed_turns = sum(s["stats"]["failed_turns"] for s in all_session_stats)
    total_turns = total_successful_turns + total_failed_turns

    if total_turns > 0:
        overall_success_rate = total_successful_turns / total_turns
        print(f"📊 Overall Statistics:")
        print(f"   🏁 Total sessions: {total_sessions}")
        print(f"   🔄 Total turns: {total_turns}")
        print(f"   ✅ Successful turns: {total_successful_turns}")
        print(f"   ❌ Failed turns: {total_failed_turns}")
        print(f"   📈 Overall success rate: {overall_success_rate:.1%}")

    # Ensure all data is sent to Langfuse
    langfuse.flush()

    print(f"\n Monitoring test completed!")
    print(f"\n Check your Langfuse dashboard for:")
    print(f"    Latency metrics and slow call alerts")
    print(f"    Error tracking and failure patterns")
    print(f"    Filter by:")
    print(f"      - is_slow_call: true (calls > {LATENCY_THRESHOLD_MS}ms)")
    print(f"      - has_error: true (failed calls)")
    print(f"      - performance_tier: slow/normal/fast")
    print(f"      - error_type: rate_limit/timeout/etc.")
    print(f"\n To set up alerts in Langfuse dashboard:")
    print(f"   1. Go to your project settings")
    print(f"   2. Create alerts for:")
    print(f"      - Latency > {LATENCY_THRESHOLD_MS}ms")
    print(f"      - Error rate > 10%")
    print(f"      - Failed generations")


if __name__ == "__main__":
    main()
