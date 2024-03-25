FORUM_RULES = """
You are going to get an instruction, here is some reference that you might need to fulfill the instruction:
------------------------------------------------------------------------------------
What is CMV?
CMV is a subreddit dedicated to civil discourse, and is built around the idea that in order to resolve our differences, we must first understand them. We believe that productive conversation requires respect and openness, and that certitude is the enemy of understanding.

That's why CMV is the perfect place to post an opinion you're open to changing. We're not looking to host aggressive debates, or encourage judgement, but help each other understand different perspectives.

Who can post?
Anyone can post here so long as they have an open-mind and are looking to consider other perspectives.

Here are some rules of the forum:
Rules apply to the original poster (OP) and their submission only:
1. Explain the reasoning behind your view, not just what that view is.
2. You must personally hold the view and demonstrate that you are open to it changing.
3. Posts cannot express a neutral stance, a stance regarding transgender, suggest harm against a specific person, be self-promotional, or discuss this subreddit.
4. Only post if you are willing to have a conversation with those who reply to you.
Rules apply to all commenters involved in the discussion:
1. Direct responses to a submission must challenge or question at least one aspect of the submitted view. Arguments in favor of the view OP is willing to change must be restricted to replies to comments.
2. Don't be rude or hostile to other users. Your comment will be removed even if the rest of it is solid.
3. Refrain from accusing OP or anyone else of being unwilling to change their view.
4. Responses must contribute meaningfully to the conversation.
------------------------------------------------------------------------------------
"""


def find_path_to_root(node_id, df):
    path_nodes = [node_id]
    parent_id = df.loc[node_id]["parent"]
    while parent_id != -1:
        path_nodes.append(parent_id)
        parent_id = df.loc[parent_id]["parent"]

    return path_nodes


def add_moderation_branch_path(conversation_df, only_bad_tone_parents=False, remove_bad_tone_parents=False):
    if only_bad_tone_parents and remove_bad_tone_parents:
        raise ValueError("Can't include only bad tone and also remove them")
    # Obtain the moderation nodes
    df_only_moderation = conversation_df[conversation_df["Moderation"] == 1]
    if only_bad_tone_parents:
        df_moderation_parent = conversation_df.loc[df_only_moderation["parent"].values]
        moderation_with_specific_parent_indices = df_moderation_parent[
            (df_moderation_parent["Aggressive"] == 1) |
            (df_moderation_parent["Ridicule"] == 1) |
            (df_moderation_parent["Sarcasm"] == 1) |
            (df_moderation_parent["Complaint"] == 1)].index
        df_moderation_with_specific_parent = df_only_moderation[df_only_moderation["parent"].isin(
            moderation_with_specific_parent_indices)]
    else:
        if remove_bad_tone_parents:
            df_moderation_parent = conversation_df.loc[df_only_moderation["parent"].values]
            moderation_with_specific_parent_indices = df_moderation_parent[
                ~((df_moderation_parent["Aggressive"] == 1) |
                  (df_moderation_parent["Ridicule"] == 1) |
                  (df_moderation_parent["Sarcasm"] == 1) |
                  (df_moderation_parent["Complaint"] == 1))].index
            df_moderation_with_specific_parent = df_only_moderation[df_only_moderation["parent"].isin(
                moderation_with_specific_parent_indices)]
        else:
            df_moderation_with_specific_parent = df_only_moderation

    # Get the moderation brach path
    df_moderation_with_specific_parent["branch path"] = df_moderation_with_specific_parent.apply(
        lambda row: find_path_to_root(row["index"], conversation_df), axis=1)

    return df_moderation_with_specific_parent


def generate_branch_prompt(moderation_df, conversation_df, moderation_index):
    branch_path = moderation_df.iloc[moderation_index]["branch path"] # of format [Moderation node, Parent, ..., OP]
    parent_node = conversation_df.loc[branch_path[1]]

    prompt = \
        """<s>[INST]\n Instruction: Assume you are a forum moderator. Please respond to {}, as his response might require a moderation note.\n\n""".format(parent_node["author"])
    for path_i in branch_path[:0:-1]:
        node_i = conversation_df.loc[path_i]
        prompt += \
            """{}: {}\n\n""".format(node_i["author"], node_i["text"])
    prompt += """Instruction: Assume you are a forum moderator. Please respond to {}, as his response might require a moderation note. Please try not to include your opinion. Be short!\n[/INST]""".format(parent_node["author"])

    return prompt, moderation_df.iloc[moderation_index]["text"]


def add_negative_branch_path(conversation_df):
    df_only_bad_tone = conversation_df[
        (conversation_df["Aggressive"] == 1) |
        (conversation_df["Ridicule"] == 1) |
        (conversation_df["Sarcasm"] == 1) |
        (conversation_df["Complaint"] == 1)]

    # Get the moderation brach path
    df_only_bad_tone["neg branch path"] = df_only_bad_tone.apply(
        lambda row: find_path_to_root(row["index"], conversation_df), axis=1)

    return df_only_bad_tone


def generate_branch_for_negative_tone_prompt(df_only_bad_tone, conversation_df,
                                             node_index, with_fourm_ruls=True,
                                             context_len=None, branch_col_name="neg branch path"):
    branch_path = df_only_bad_tone.iloc[
        node_index][branch_col_name]  # of format [Negative tone node, Parent, ..., OP]
    neg_tone_node = conversation_df.loc[branch_path[0]]

    op_index = branch_path[-1]
    # in case we want the OP message and only some of the last messages
    if context_len is not None:
        context_len = min(len(branch_path) - 1, context_len)
        branch_path = branch_path[:context_len]
    else:
        branch_path = branch_path[:-1]

    # generate the prompt
    prompt = FORUM_RULES if with_fourm_ruls else ""
    prompt += \
        """Instruction: Assume you are a forum moderator. Please respond to {}, as his response might require a moderation note. Pay attention to the context and converstion subject and flow.\n\n""".format(neg_tone_node["author"])
    op_node = conversation_df.loc[op_index]
    prompt += \
        """{}: {}\n\n""".format(op_node["author"], op_node["text"])

    if context_len is not None:
        prompt += "---- Hidden part of the converstion ----\n\n"

    for path_i in branch_path[::-1]:
        node_i = conversation_df.loc[path_i]
        prompt += \
            """{}: {}\n\n""".format(node_i["author"], node_i["text"])
    prompt += """Instruction: Assume you are a forum moderator. Please respond to {}, as his response might require a moderation note. Please try not to include your opinion. Pay attention to the context and the converstion subject and flow. Be concise but also informative!\n Be kind if possible and natural like humans. Avoid messages that starts with "Moderation Note: ..." or "Moderator: ..." or other robotic like responses. Address the user you are replying to.\n""".format(neg_tone_node["author"])

    return prompt


def generate_branch_for_negative_tone_prompt_for_mistral(df_only_bad_tone, conversation_df,
                                                         node_index, with_fourm_ruls=False,
                                                         context_len=None, branch_col_name="neg branch path"):
    branch_path = df_only_bad_tone.iloc[
        node_index][branch_col_name]  # of format [Negative tone node, Parent, ..., OP]
    neg_tone_node = conversation_df.loc[branch_path[0]]

    op_index = branch_path[-1]
    # in case we want the OP message and only some of the last messages
    if context_len is not None:
        context_len = min(len(branch_path) - 1, context_len)
        branch_path = branch_path[:context_len]
    else:
        branch_path = branch_path[:-1]

    # generate the prompt
    prompt = FORUM_RULES if with_fourm_ruls else ""
    prompt += \
        """<s>[INST] Instruction: Assume you are a forum moderator. Please respond to {}, as his response might require a moderation note.\n\n""".format(neg_tone_node["author"])
    op_node = conversation_df.loc[op_index]
    prompt += \
        """{}: {}\n\n""".format(op_node["author"], op_node["text"])

    if context_len is not None:
        prompt += "---- Hidden part of the converstion ----\n\n"

    for path_i in branch_path[::-1]:
        node_i = conversation_df.loc[path_i]
        prompt += \
            """{}: {}\n\n""".format(node_i["author"], node_i["text"])
    prompt += """Instruction: Assume you are a forum moderator. Please respond to {}, as his response might require a moderation note. Please try not to include your opinion. Pay attention to the context and the converstion subject and flow. Be concise but also informative!\n Be kind if possible and natural like humans. Avoid messages that starts with "Moderation Note: ..." or "Moderator: ..." or other robotic like responses. Address the user you are replying to.\n [/INST]""".format(neg_tone_node["author"])

    return prompt


def obtain_pseudo_positive_conversation(conversation_df):
    pseudo_positive_df = conversation_df[
        (
            (conversation_df["Moderation"]) |
            (conversation_df["RequestClarification"]) |
            (conversation_df["AttackValidity"]) |
            (conversation_df["Clarification"]) |
            (conversation_df["Answer"]) |
            (conversation_df["CounterArgument"]) |
            (conversation_df["Extension"]) |
            (conversation_df["ViableTransformation"]) |
            (conversation_df["Personal"]) |
            (conversation_df["Positive"]) |
            (conversation_df["WQualifiers"]) |
            (conversation_df["Softening"]) |
            (conversation_df["AgreeBut"]) |
            (conversation_df["DoubleVoicing"]) |
            (conversation_df["Sources"]) 
        ) &
        (~(
            (conversation_df["BAD"]) |
            (conversation_df["Repetition"]) |
            (conversation_df["NegTransformation"]) |
            (conversation_df["NoReasonDisagreement"]) |
            (conversation_df["Convergence"]) |
            (conversation_df["AgreeToDisagree"]) |
            (conversation_df["Aggressive"]) |
            (conversation_df["Complaint"]) |
            (conversation_df["Sarcasm"]) |
            (conversation_df["RephraseAttack"]) |
            (conversation_df["CriticalQuestion"]) |
            (conversation_df["Alternative"]) |
            (conversation_df["DirectNo"]) |
            (conversation_df["Irrelevance"]) |
            (conversation_df["Nitpicking"])
        ))
        ]
    # add branch
    pseudo_positive_df["pseudo positive branch path"] = pseudo_positive_df.apply(
        lambda row: find_path_to_root(row["index"], conversation_df), axis=1)

    return pseudo_positive_df
