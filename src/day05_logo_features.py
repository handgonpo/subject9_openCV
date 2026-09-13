from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. 이미지 읽기
# --------------------------------------------------

def read_image(path):

    path = Path(path)

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise FileNotFoundError(
            f"이미지를 읽지 못했습니다: {path}"
        )

    return image


# --------------------------------------------------
# 2. Reference 준비
# --------------------------------------------------

def prepare_reference(reference_path):

    reference = read_image(
        reference_path
    )

    reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )


    # SIFT를 사용할 수 있는지 확인
    if not hasattr(
        cv2,
        "SIFT_create"
    ):
        raise RuntimeError(
            "현재 OpenCV에서 SIFT를 사용할 수 없습니다."
        )


    # SIFT 객체 생성
    sift = cv2.SIFT_create(
        nfeatures=1500
    )


    # 기준 로고의 특징점과 Descriptor 생성
    ref_keypoints, ref_descriptors = (
        sift.detectAndCompute(
            reference_gray,
            None
        )
    )


    if (
        ref_descriptors is None
        or len(ref_keypoints) < 4
    ):
        raise RuntimeError(
            "Reference에서 충분한 SIFT 특징점을 찾지 못했습니다."
        )


    return {
        "image": reference,
        "gray": reference_gray,
        "sift": sift,
        "keypoints": ref_keypoints,
        "descriptors": ref_descriptors,
        "width": reference.shape[1],
        "height": reference.shape[0],
    }


# --------------------------------------------------
# 3. Target 한 장 측정
# --------------------------------------------------

def measure_target(
    target_path,
    reference_data
):

    target = read_image(
        target_path
    )

    target_gray = cv2.cvtColor(
        target,
        cv2.COLOR_BGR2GRAY
    )


    # Reference에서 미리 준비한 값 가져오기
    reference_gray = (
        reference_data["gray"]
    )

    ref_keypoints = (
        reference_data["keypoints"]
    )

    ref_descriptors = (
        reference_data["descriptors"]
    )

    sift = (
        reference_data["sift"]
    )


    # ==================================================
    # 1. Template Matching
    # ==================================================

    template_result = cv2.matchTemplate(
        target_gray,
        reference_gray,
        cv2.TM_CCOEFF_NORMED
    )


    (
        _,
        template_score,
        _,
        template_location
    ) = cv2.minMaxLoc(
        template_result
    )


    # ==================================================
    # 2. Target SIFT 특징점
    # ==================================================

    target_keypoints, target_descriptors = (
        sift.detectAndCompute(
            target_gray,
            None
        )
    )


    good_matches = []


    if target_descriptors is not None:

        matcher = cv2.BFMatcher(
            cv2.NORM_L2
        )


        knn_matches = matcher.knnMatch(
            ref_descriptors,
            target_descriptors,
            k=2
        )


        # Ratio Test
        for pair in knn_matches:

            if len(pair) < 2:
                continue


            m, n = pair


            if (
                m.distance
                < 0.75 * n.distance
            ):

                good_matches.append(
                    m
                )


    # ==================================================
    # 3. RANSAC + Homography
    # ==================================================

    homography_found = 0

    inlier_count = 0

    inlier_ratio = 0.0

    polygon = None


    if len(good_matches) >= 4:

        source_points = np.float32(
            [
                ref_keypoints[
                    match.queryIdx
                ].pt
                for match in good_matches
            ]
        ).reshape(
            -1,
            1,
            2
        )


        target_points = np.float32(
            [
                target_keypoints[
                    match.trainIdx
                ].pt
                for match in good_matches
            ]
        ).reshape(
            -1,
            1,
            2
        )


        homography, mask = (
            cv2.findHomography(
                source_points,
                target_points,
                cv2.RANSAC,
                5.0
            )
        )


        if (
            homography is not None
            and mask is not None
        ):

            homography_found = 1


            # RANSAC을 통과한 Match 개수
            inlier_count = int(
                mask.ravel().sum()
            )


            # Good Match 중 Inlier 비율
            inlier_ratio = (
                inlier_count
                / len(good_matches)
            )


            # ------------------------------------------
            # Reference 로고 영역을 Target에 투영
            # ------------------------------------------

            ref_width = (
                reference_data["width"]
            )

            ref_height = (
                reference_data["height"]
            )


            corners = np.float32(
                [
                    [0, 0],

                    [
                        ref_width - 1,
                        0
                    ],

                    [
                        ref_width - 1,
                        ref_height - 1
                    ],

                    [
                        0,
                        ref_height - 1
                    ],
                ]
            ).reshape(
                -1,
                1,
                2
            )


            projected = (
                cv2.perspectiveTransform(
                    corners,
                    homography
                )
            )


            if np.isfinite(
                projected
            ).all():

                polygon = (
                    projected
                    .reshape(-1, 2)
                )


    # ==================================================
    # 측정 결과 반환
    # ==================================================

    return {

        "target_path":
            str(target_path),

        "template_score":
            float(template_score),

        "template_x":
            int(
                template_location[0]
            ),

        "template_y":
            int(
                template_location[1]
            ),

        "sift_good_matches":
            int(
                len(good_matches)
            ),

        "homography_found":
            int(
                homography_found
            ),

        "homography_inliers":
            int(
                inlier_count
            ),

        "inlier_ratio":
            float(
                inlier_ratio
            ),

        "polygon":
            polygon,

        "target_image":
            target,
    }