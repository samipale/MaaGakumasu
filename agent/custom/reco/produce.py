import time
from utils import logger
from collections import Counter
from typing import Union, Optional

from maa.define import RectType
from maa.context import Context
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition

# 以下为开发功能使用，不能上传至github
import os
import numpy as np
from PIL import Image
from datetime import datetime

def save_train_data(image):
    current_time = datetime.now()
    save_path = r'D:\scripts\MFAAvalonia-v2.4.0-win-x64\debug\train'
    timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # %f是微秒，取前3位得到毫秒
    file_base = f"options-{timestamp}.png"
    filename = os.path.join(save_path, file_base)
    height, width, _ = image.shape
    rgb_array = image[:, :, ::-1]
    alpha_channel = np.full((height, width, 1), 255, dtype=np.uint8)
    rgba_array = np.concatenate([rgb_array, alpha_channel], axis=2)
    img = Image.fromarray(rgba_array, mode='RGBA')
    img.save(filename)
# 以上为开发功能使用，不能上传至github

@AgentServer.custom_recognition("ProduceShowStart")
class ProduceShowStart(CustomRecognition):
    """
        检测通过屏幕是否旋转判断演出开始
    """

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        image = argv.image
        height = image.shape[0]
        width = image.shape[1]
        context.run_task("Click_1")
        if height < width:
            logger.success("事件: 演出开始")
            return CustomRecognition.AnalyzeResult(box=[0, 0, 0, 0], detail={"detail": "屏幕旋转"})
        return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "屏幕未旋转"})


@AgentServer.custom_recognition("ProduceShowEnd")
class ProduceShowEnd(CustomRecognition):
    """
        检测屏幕是否旋转
    """

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        image = argv.image
        height = image.shape[0]
        width = image.shape[1]
        context.run_task("Click_1")
        if height > width:
            logger.success("事件: 演出结束")
            return CustomRecognition.AnalyzeResult(box=[0, 0, 0, 0], detail={"detail": "屏幕旋转"})
        return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "屏幕未旋转"})


@AgentServer.custom_recognition("ProduceChooseCardsAuto")
class ProduceChooseCardsAuto(CustomRecognition):
    """
        自动识别选择卡牌
        优先选择活动与推荐，没有则根据培育对象选择
    """

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        reco_detail_recommend = context.run_recognition(
            "ProduceChooseCardsSuggestion", argv.image,
            pipeline_override={"ProduceChooseCardsSuggestion": {
                "recognition": "TemplateMatch",
                "template": "produce/recommed.png",
                "roi": [86, 788, 545, 220]
            }})
        reco_detail_event = context.run_recognition(
            "ProduceChooseEventCards", argv.image,
            pipeline_override={"ProduceChooseEventCards": {
                "recognition": "TemplateMatch",
                "template": "produce/event_recommend.png",
                "roi": [86, 788, 545, 220]
            }})

        logger.success("事件: 选择卡牌")
        if reco_detail_recommend.hit:
            logger.info("选择建议卡")
            result = reco_detail_recommend.best_result.box
            result[1] = result[1] - 80
            return CustomRecognition.AnalyzeResult(box=result, detail={"detail": "选择建议卡"})
        elif reco_detail_event.hit:
            logger.info("选择活动卡")
            result = reco_detail_event.best_result.box
            result[1] = result[1] + 80
            return CustomRecognition.AnalyzeResult(box=result, detail={"detail": "选择活动卡"})
        else:
            logger.info("选择第一张卡")
            result = [160, 824, 20, 20]
            return CustomRecognition.AnalyzeResult(box=result, detail={"detail": "选择第一张卡"})


@AgentServer.custom_recognition("ProduceChooseDrinkAuto")
class ProduceChooseDrinkAuto(CustomRecognition):
    """
        自动识别选择饮料
        优先选择第一个
    """

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        reco_detail = context.run_recognition(
            "ProduceChooseDrinkFull", argv.image,
            pipeline_override={"ProduceChooseDrinkFull": {
                "recognition": "TemplateMatch",
                "template": "produce/drink_reject.png",
                "roi": [54, 950, 610, 98]
            }})
        logger.success("事件: 选择饮料")
        if reco_detail and reco_detail.hit:
            logger.info("放弃饮料")
            return CustomRecognition.AnalyzeResult(box=reco_detail.best_result.box, detail={"detail": "放弃饮料"})
        else:
            logger.info("选择第一个饮料")
            result = [160, 824, 124, 124]
            return CustomRecognition.AnalyzeResult(box=result, detail={"detail": "选择第一个饮料"})


@AgentServer.custom_recognition("ProduceChooseItemAuto")
class ProduceChooseItemAuto(CustomRecognition):
    """
        自动识别选择饮料
        优先选择第一个
    """

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        logger.success("事件: 选择物品")
        logger.info("选择第一个物品")
        result = [160, 824, 124, 124]
        return CustomRecognition.AnalyzeResult(box=result, detail={"detail": "选择第一个物品"})


@AgentServer.custom_recognition("ProduceCardsFlagAuto")
class ProduceCardsFlagAuto(CustomRecognition):
    """
        自动识别出牌场景
    """

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:

        context.run_task("Click_1")
        cards_reco_detail = context.run_recognition("ProduceRecognitionCards", argv.image)
        move_cards_reco_detail = context.run_recognition("ProduceRecognitionMoveCards", argv.image)
        # 通过识别卡牌与生命值判断是否进入出牌场景，将hp判断放进if后面减少ProduceEntry节点压力
        if cards_reco_detail.hit and context.run_recognition("ProduceRecognitionHealthFlag", argv.image).hit:
            logger.success("事件: 出牌场景")
            return CustomRecognition.AnalyzeResult(box=[0, 0, 0, 0], detail={"detail": "识别到出牌场景"})
        elif move_cards_reco_detail.hit:
            logger.success("事件: 出牌场景")
            return CustomRecognition.AnalyzeResult(box=[0, 0, 0, 0], detail={"detail": "识别到出牌场景（移动卡牌界面）"})
        else:
            return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "未识别到选择场景"})


@AgentServer.custom_recognition("ProduceOptionsFlagAuto")
class ProduceOptionsFlagAuto(CustomRecognition):
    """
        自动识别选择冲刺/上课/外出场景
        冲刺识别建议需要在设置开启培育指引
    """

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        context.run_task("Click_1")
        # 分两段判断，减轻ProduceEntry节点压力
        options_reco_detail = context.run_recognition("ProduceRecognitionOptions", argv.image)
        if not options_reco_detail.hit:
            return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "未识别到选择冲刺/上课/外出场景"})
        if context.run_recognition("ProduceRecognitionPushEvent", argv.image).hit:
            event = "Push"
        elif context.run_recognition("ProduceRecognitionLessonEvent", argv.image).hit:
            event = "Lesson"
        elif context.run_recognition("ProduceRecognitionGoOutEvent", argv.image).hit:
            event = "GoOut"
        elif context.run_recognition("ProduceRecognitionWeekEvent", argv.image).hit:
            # 以下为开发功能，不要上传至github
            save_train_data(argv.image)
            # 以上为开发功能，不要上传至github
            logger.success("事件: 开局回忆会话")
            result = options_reco_detail.best_result.box
            context.tasker.controller.post_click(result[0] + int(result[2] / 2),
                                                 result[1] + int(result[3] / 2)).wait()
            time.sleep(0.2)
            context.tasker.controller.post_click(result[0] + int(result[2] / 2),
                                                 result[1] + int(result[3] / 2)).wait()
            return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "会话场景"})
        else:
            return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "未识别到会话场景"})

        # 成功进入场景，重新截图
        if event == "Push":
            logger.success("事件: 冲刺")
            time.sleep(3)
            image = context.tasker.controller.post_screencap().wait().get()
        else:
            image = argv.image

        # 开始执行选项判断
        if event == "Push":
            # logger.success("事件: 冲刺")
            reco_detail = context.run_recognition(
                "ProduceGetPushSuggestion", image,
                pipeline_override={"ProduceGetPushSuggestion": {
                    "recognition": "OCR",
                    "expected": ["ボーカル", "ダンス", "ジュアル"],
                    "roi": [270, 160, 350, 80]
                }}
            )

            # if reco_detail.all_results:
            #     suggestion_text = "".join(item.text for item in reco_detail.all_results)
            #     logger.info(f"ocr检测结果: {suggestion_text}")

            template_filename = "push_vi.png"
            if reco_detail.hit:
                suggestion_text = "".join(item.text for item in reco_detail.all_results)
                logger.info(f"老师建议: {suggestion_text}")
                if "ボーカル" in suggestion_text:
                    template_filename = "push_vo.png"
                elif "ダンス" in suggestion_text:
                    template_filename = "push_da.png"
                elif "ジュアル" in suggestion_text:
                    template_filename = "push_vi.png"
            else:
                logger.info(f"识别老师建议失败，默认选择视觉")

            reco_detail = context.run_recognition(
                "ProduceChoosePushOption", image,
                pipeline_override={"ProduceChoosePushOption": {
                    "recognition": "TemplateMatch",
                    "template": f"produce/{template_filename}",
                    "roi": [0, 600, 720, 350]
                }}
            )

            # 输出冲刺选项
            if reco_detail.hit:
                # 以下为开发功能，不要上传至github
                save_train_data(argv.image)
                # 以上为开发功能，不要上传至github
                result = reco_detail.best_result.box
                context.tasker.controller.post_click(result[0] + int(result[2] / 2),
                                                     result[1] + int(result[3] / 2)).wait()
                time.sleep(0.2)
                context.tasker.controller.post_click(result[0] + int(result[2] / 2),
                                                     result[1] + int(result[3] / 2)).wait()
                return CustomRecognition.AnalyzeResult(box=result, detail={"detail": "识别到冲刺选项"})
            else:
                logger.warning(f"未识别到冲刺选项")
                return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "未识别到冲刺选项"})

        elif event == "Lesson":
            logger.success("事件: 上课")
            # 体力课程，默认选择体力-4的选项
            reco_detail = context.run_recognition(
                "ProduceChooseLessonOption", image,
                pipeline_override={"ProduceChooseLessonOption": {
                    "recognition": "TemplateMatch",
                    "template": "produce/lesson_health-4.png",
                    "roi": [0, 600, 720, 350],
                    "threshold": 0.9
                }}
            )
            if reco_detail.hit:
                # 以下为开发功能，不要上传至github
                save_train_data(argv.image)
                # 以上为开发功能，不要上传至github
                logger.success("选择消耗4体力的选项")
                result = reco_detail.best_result.box
                result[0] = result[0] - 200
                result[1] = result[1] + 50
                context.tasker.controller.post_click(result[0], result[1]).wait()
                time.sleep(0.2)
                context.tasker.controller.post_click(result[0], result[1]).wait()
                return CustomRecognition.AnalyzeResult(box=result, detail={"detail": "识别到上课选项"})

            # 突入普通课程（出牌模式）
            reco_detail = context.run_recognition(
                "ProduceRecognitionNormalLesson", image,
                pipeline_override={"ProduceRecognitionNormalLesson": {
                    "recognition": "TemplateMatch",
                    "template": ["produce/lesson_vo.png", "produce/lesson_da.png", "produce/lesson_vi.png"],
                    "roi": [0, 600, 720, 350],
                    "threshold": 0.9
                }}
            )
            if reco_detail.hit:
                # 以下为开发功能，不要上传至github
                save_train_data(argv.image)
                # 以上为开发功能，不要上传至github
                logger.success("普通课程（出牌模式），随机选择")
                result = reco_detail.best_result.box
                context.tasker.controller.post_click(result[0] + int(result[2] / 2),
                                                     result[1] + int(result[3] / 2)).wait()
                time.sleep(0.2)
                context.tasker.controller.post_click(result[0] + int(result[2] / 2),
                                                     result[1] + int(result[3] / 2)).wait()
                return CustomRecognition.AnalyzeResult(box=result, detail={"detail": "识别到上课选项"})

            # 其他情况
            logger.warning("未识别到上课选项")
            return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "未识别到上课选项"})

        elif event == "GoOut":
            logger.success("事件: 外出")
            reco_detail = context.run_recognition("ProduceRecognitionPoint", image)
            if reco_detail.hit:
                try:
                    points = int(reco_detail.best_result.text.replace(",", ""))
                    logger.info(f"积分: {points}")
                except ValueError:
                    logger.warning("积分数据解析失败")
                    points = 0
            else:
                logger.warning("积分数据解析失败")
                points = 0

            result = None
            go_out_log = "外出选项判断出现问题"
            if points >= 100:
                # result = [360, 700, 0, 0]
                reco_detail = context.run_recognition(
                    "ProduceChooseLessonOption", image,
                    pipeline_override={"ProduceChooseLessonOption": {
                        "recognition": "TemplateMatch",
                        "template": f"produce/go_out_p-100.png",
                        "roi": [0, 600, 720, 350],
                        "threshold": 0.9
                    }}
                )
                if reco_detail.hit:
                    go_out_log = "选择消耗100积分的选项"
                    result = reco_detail.best_result.box
                    result[0] = result[0] - 200
                    result[1] = result[1] + 50

            if points >= 50 and result is None:
                # result = [360, 800, 0, 0]
                reco_detail = context.run_recognition(
                    "ProduceChooseLessonOption", image,
                    pipeline_override={"ProduceChooseLessonOption": {
                        "recognition": "TemplateMatch",
                        "template": f"produce/go_out_p-50.png",
                        "roi": [0, 600, 720, 350],
                        "threshold": 0.9
                    }}
                )
                if reco_detail.hit:
                    go_out_log = "选择消耗50积分的选项"
                    result = reco_detail.best_result.box
                    result[0] = result[0] - 200
                    result[1] = result[1] + 50

            if points < 50 and result is None:
                go_out_log = "选择消耗0积分的选项"
                result = [360, 900, 0, 0]

            # 输出外出选项
            if result:
                # 以下为开发功能，不要上传至github
                save_train_data(argv.image)
                # 以上为开发功能，不要上传至github
                context.tasker.controller.post_click(result[0], result[1]).wait()
                time.sleep(0.2)
                context.tasker.controller.post_click(result[0], result[1]).wait()
                logger.success(go_out_log)
                return CustomRecognition.AnalyzeResult(box=result, detail={"detail": "识别到外出选项"})
            else:
                logger.warning("未识别到外出选项")
                return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "未识别到外出选项"})

        else:
            return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "未识别到冲刺/上课/外出选项"})

        # logger.success("事件: 选择冲刺/上课/外出")
        # results = options_reco_detail.filtered_results
        # label_counts = Counter()
        # best_choose = [0, 1279, 0, 0]
        # for result in results:
        #     label_counts[result.label] += 1
        #     if result.label == "choose":
        #         if result.box[1] < best_choose[1]:
        #             best_choose = result.box
        # choose = label_counts["choose"]
        # lesson = label_counts["lesson"]
        # logger.info(f"选项数量:{choose}/{lesson}")
        # if choose > lesson:
        #     best_box = best_choose
        # else:
        #     best_box = options_reco_detail.best_result.box
        # if lesson == 0:
        #     print("")
        #
        # context.tasker.controller.post_click(best_box[0]+int(best_box[2]/2), best_box[1]+int(best_box[3]/2)).wait()
        # return CustomRecognition.AnalyzeResult(box=best_box, detail={"detail": "选择加最佳选项"})
