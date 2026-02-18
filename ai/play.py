from ai.utils import FixedRuntime, get_models, get_validated_input, EModelType
from ai.eval import ActionsThread, AimThread, CombinedThread
import traceback


def start_play(eval_key: str = '\\'):
    try:
        action_models = get_models(EModelType.Actions)
        aim_models = get_models(EModelType.Aim)
        combined_models = get_models(EModelType.Combined)

        user_choice = get_validated_input(f"""What type of model would you like to test?
    [0] Aim Model | {len(aim_models)} Available
    [1] Actions Model | {len(action_models)} Available
    [2] Combined Model | {len(combined_models)} Available
    [3] Dual Mode (Aim + Actions) | {len(aim_models)} Aim, {len(action_models)} Actions
""", lambda a: a.strip().isnumeric() and (0 <= int(a.strip()) <= 3), lambda a: int(a.strip()))

        if user_choice == 3:
            # --- 雙線程模式：Aim + Actions 同時運行 ---
            if not aim_models:
                print("No Aim models available! Train one first.")
                return
            if not action_models:
                print("No Actions models available! Train one first.")
                return

            # 選擇 Aim 模型
            prompt = "Select Aim model (mouse control):\n"
            for i in range(len(aim_models)):
                prompt += f"    [{i}] {aim_models[i]}\n"
            aim_idx = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                    0 <= int(a.strip()) < len(aim_models)), lambda a: int(a.strip()))

            # 選擇 Actions 模型
            prompt = "Select Actions model (key control):\n"
            for i in range(len(action_models)):
                prompt += f"    [{i}] {action_models[i]}\n"
            action_idx = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                    0 <= int(a.strip()) < len(action_models)), lambda a: int(a.strip()))

            # 兩個線程用同一個 eval_key 控制
            aim_thread = AimThread(model_id=aim_models[aim_idx]['id'], eval_key=eval_key)
            action_thread = ActionsThread(model_id=action_models[action_idx]['id'], eval_key=eval_key)

            aim_thread.start()
            action_thread.start()
            print(f"\n[Dual Mode] Aim + Actions running. Press '{eval_key}' to toggle both.")

        else:
            active_model = None
            if user_choice == 0:
                prompt = "What aim model would you like to use?\n"
                for i in range(len(aim_models)):
                    prompt += f"    [{i}] {aim_models[i]}\n"

                model_index = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                        0 <= int(a.strip()) < len(aim_models)), lambda a: int(a.strip()))

                active_model = AimThread(model_id=aim_models[model_index]['id'], eval_key=eval_key)

            elif user_choice == 1:
                prompt = "What actions model would you like to use?\n"
                for i in range(len(action_models)):
                    prompt += f"    [{i}] {action_models[i]}\n"

                model_index = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                        0 <= int(a.strip()) < len(action_models)), lambda a: int(a.strip()))

                active_model = ActionsThread(
                    model_id=action_models[model_index]['id'], eval_key=eval_key)
            else:
                prompt = "What combined model would you like to use?\n"
                for i in range(len(combined_models)):
                    prompt += f"    [{i}] {combined_models[i]}\n"

                model_index = get_validated_input(prompt, lambda a: a.strip().isnumeric() and (
                        0 <= int(a.strip()) < len(combined_models)), lambda a: int(a.strip()))

                active_model = CombinedThread(
                    model_id=combined_models[model_index]['id'], eval_key=eval_key)

            if active_model:
                active_model.start()

        try:
            input("\nPress Enter to quit.\n")
        except KeyboardInterrupt:
            pass
    except Exception as e:
        traceback.print_exc()
