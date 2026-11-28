// include/strata/models/common/card_stage.hpp - per-card execution for the Qwen3.8 and DeepSeek V4.1 paged models.
// The model prepares and finishes each whole group; CardStage runs one card's layers. CardPipelineModel adapts
// these stages to PagedModel's pipeline calls and waits for their state writes before pages can be reused.
#pragma once

#include "strata/common/serve/paged_model.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::models {

/// One card's share of a pipelined decode step: the card's layers for a group of rows its model has prepared.
class CardStage {
public:
    virtual ~CardStage() = default;
    /// Queues this card's layers for prepared group `group` on the card's own stream and returns without waiting.
    virtual bool launch(int group, std::string& err) = 0;
    /// Gets 1 once this card has finished the group it was given, 0 while it is busy, or -1 (with `err`) on
    /// failure. Meanwhile it may do work the GPU hands to the host (Qwen3.8's experts on the CPU).
    virtual int poll(std::string& err) = 0;
    /// Waits until this card's writes to conversation state that are in flight are done.
    virtual bool wait_writes(std::string& err) = 0;
};

/// A PagedModel whose pipeline stages are its cards: launch_group, poll_group and wait_state go to the stage's
/// CardStage. The family fills `stages_` in card order before serving.
class CardPipelineModel : public strata::serve::PagedModel {
public:
    int get_pipeline_stages() const override { return can_pipeline() ? (int) stages_.size() : 0; }
    bool launch_group(int stage, int group, std::string& err) override {
        return is_group_idle(group) || stages_[(size_t) stage]->launch(group, err);
    }
    int poll_group(int stage, int group, std::vector<int32_t>&, std::string& err) override {
        return is_group_idle(group) ? 1 : stages_[(size_t) stage]->poll(err);
    }
    /// Waits for every card's state writes, also after one failed, so that no page is reused while a write to it
    /// may still run; `err` gets the first failure.
    bool wait_state(std::string& err) override {
        bool ok = true;
        for (const std::unique_ptr<CardStage>& s : stages_) {
            std::string why;
            if (!s->wait_writes(why) && ok) {
                ok = false;
                err = why;
            }
        }
        return ok;
    }

protected:
    /// Tells whether groups may overlap across cards; false makes get_pipeline_stages report zero.
    virtual bool can_pipeline() const { return true; }
    /// Gets whether prepared group `group` has nothing for the cards to run, so that its stages finish at once.
    virtual bool is_group_idle(int) const { return false; }

    /// Owns the per-card adapters in execution order; their referenced resources stay valid while serving.
    std::vector<std::unique_ptr<CardStage>> stages_;
};

}  // namespace strata::models
