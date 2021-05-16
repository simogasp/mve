/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "util/timer.h"
#include "mve/image.h"
#include "mve/image_exif.h"
#include "mve/image_tools.h"
#include "sfm/bundler_common.h"
#include "sfm/extract_focal_length.h"
#include "sfm/bundler_features.h"

SFM_NAMESPACE_BEGIN
SFM_BUNDLER_NAMESPACE_BEGIN

void
Features::compute (mve::Scene::Ptr scene, ViewportList* viewports)
{
    if (scene == nullptr)
        throw std::invalid_argument("Null scene given");
    if (viewports == nullptr)
        throw std::invalid_argument("No viewports given");

    mve::Scene::ViewList const& views = scene->get_views();

    /* Initialize viewports. */
    viewports->clear();
    viewports->resize(views.size());

    std::size_t num_views = viewports->size();
    std::size_t num_done = 0;
    std::size_t total_features = 0;

    /* Iterate the scene and compute features. */
#pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < static_cast<int>(views.size()); ++i)
    {
#pragma omp critical
        {
            num_done += 1;
            float percent = (num_done * 1000 / num_views) / 10.0f;
            std::cout << "\rDetecting features, view " << num_done << " of "
                << num_views << " (" << percent << "%)..." << std::flush;
        }

        if (views[i] == nullptr)
            continue;

        mve::View::Ptr view = views[i];
        mve::ByteImage::Ptr image = view->get_byte_image
            (this->opts.image_embedding);
        if (image == nullptr)
            continue;

        /* Rescale image until maximum image size is met. */
        util::WallTimer timer;
        while (this->opts.max_image_size > 0
            && image->width() * image->height() > this->opts.max_image_size)
            image = mve::image::rescale_half_size<uint8_t>(image);

        /* Compute features for view. */
        Viewport* viewport = &viewports->at(i);
        viewport->features.set_options(this->opts.feature_options);
        viewport->features.compute_features(image);
        viewport->features.normalize_feature_positions(
            viewport->principal_point[0], viewport->principal_point[1]);

#pragma omp critical
        {
            std::size_t const num_feats = viewport->features.positions.size();
            std::cout << "\rView ID "
                << util::string::get_filled(view->get_id(), 4, '0') << " ("
                << image->width() << "x" << image->height() << "), "
                << util::string::get_filled(num_feats, 5, ' ') << " features"
                << ", took " << timer.get_elapsed() << " ms." << std::endl;
            total_features += viewport->features.positions.size();
        }

        /* Clean up unused embeddings. */
        image.reset();
        view->cache_cleanup();
    }

    std::cout << "\rComputed " << total_features << " features "
        << "for " << num_views << " views (average "
        << (total_features / num_views) << ")." << std::endl;
}

SFM_BUNDLER_NAMESPACE_END
SFM_NAMESPACE_END
