use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;

mod initial_guess;
mod matrix_completion_gd;
mod matrix_completion_svd;
mod matrix_factorization_als;
mod matrix_factorization_svd;
mod print_array;

use discorec::Dataset;
use discorec::RecommenderBuilder;
use itertools::Itertools;
use ndarray::{Array1, Array2};

use ndarray::s;
use ndarray::Axis;
use ndarray_linalg::Norm;
use std::f64;

use crate::print_array::print_array;
use crate::print_array::print_array32;
use rusqlite::{params, Connection, Result, Statement};
use rustc_hash::FxHashMap;

// TODO: after one item changed someone's mind, other items which the user voted on right ifter
// have a similar context signature. How do we figure out which item actually changed the mind, so that no change is detected for the other items. Can we look at the target items in chronological order and detect patterns?
const AGREE: bool = true; // agree or helpfulnessLevel
const MAX_TARGET_ITEMS: usize = 1000;
const MAX_CONTEXT_SIZE: usize = 20; // other votes from user before and after voting on target post
const MIN_USERS_VOTED_ON_TARGET_ITEM: usize = 6;
const MAX_USERS_VOTED_ON_TARGET_ITEM: usize = 6;
const MIN_CONTEXT_VOTES_BEFORE_AND_AFTER: usize = 10;
const CONTEXT_MATRIX_FACTORIZATION_RANK: usize = 50;
const CHANGE_MATRIX_FACTORIZATION_RANK: usize = 50;

fn main() -> Result<(), Box<dyn Error>> {
    let conn = Connection::open("dataset/ratings.db")?;

    conn.execute("delete from scores", ())?;
    conn.execute("delete from user_change", ())?;
    //  where noteId = '1400247230330667008'
    //   where noteId = '1354864556552712194'
    // 10x133  where noteid = '1709553622751588430'
    // 39x544  where noteid = '1354855204005453826'
    let mut stmt_target_note_ids =
        conn.prepare("select distinct noteId, summary from ratings join notes using(noteId) order by noteId asc limit 1000000 offset 1000")?;
    let mut stmt_voters_on_note =
        conn.prepare("select raterParticipantId, createdAtMillis from ratings where noteId = ?1")?;
    let mut stmt_note_text = conn.prepare("select summary from notes where noteId = ?1")?;
    // TODO: filter users/items with only one vote
    let mut stmt_votes_before =
                    conn.prepare(if AGREE {
                "select noteId, agree - disagree from ratings where raterParticipantId = ?1 and createdAtMillis < ?2 and agree + disagree != 0 order by createdAtMillis desc limit ?3;"
                } else {"select noteId, helpfulnessLevel from ratings where raterParticipantId = ?1 and createdAtMillis < ?2 and helpfulnesslevel != '' order by createdAtMillis desc limit ?3;"})?;
    let mut stmt_votes_after =
                    conn.prepare(if AGREE {"select noteId, agree - disagree from ratings where raterParticipantId = ?1 and createdAtMillis > ?2 and agree + disagree != 0 order by createdAtMillis asc limit ?3;"} else {"select noteId, helpfulnessLevel from ratings where raterParticipantId = ?1 and createdAtMillis > ?2 and helpfulnesslevel != '' order by createdAtMillis asc limit ?3;" })?;

    // matrix where users appear two times per item: before and after voting on that item.
    // That means we get a user-item matrix for votes, where every user is a virtual user.
    // The goal is to detect change of voting behavior in the same user-embedding space.
    // target | item 0          | item 1
    //        +-----------------+------------
    //  users |1a 1b 2a 2b 3a 3b| 1a 1b 2a 2b ...
    //        +-----------------+------------
    // item 0 |
    // item 1 |              <votes>
    // item 2 |
    //          ...

    // virtual_user, context_item_id -> vote value
    let mut votes_by_virtual_users: HashMap<(VirtualUser, i64), f64, rustc_hash::FxBuildHasher> =
        FxHashMap::default();
    let mut data = Dataset::new();

    // track item_id mappings for the context matrix
    let mut context_item_idx: HashMap<i64, usize, rustc_hash::FxBuildHasher> = FxHashMap::default();

    // track VirtualUser mappings for the context matrix
    let mut virtual_user_idx: HashMap<VirtualUser, usize, rustc_hash::FxBuildHasher> =
        FxHashMap::default();

    // track target_item_id mappings for the final change-matrix
    let mut target_item_idx: HashMap<i64, usize, rustc_hash::FxBuildHasher> = FxHashMap::default();

    // track target_item_id mappings for the final change-matrix
    let mut user_idx: HashMap<String, usize, rustc_hash::FxBuildHasher> = FxHashMap::default();

    let target_item_candidate_ids_iter = stmt_target_note_ids
        .query_map(params![], |row| {
            Ok(Item {
                id: row.get::<_, i64>(0)?,
                text: row.get(1)?,
            })
        })?
        .filter_map(Result::ok);

    let mut usable_target_item_count = 0;
    for target_item in target_item_candidate_ids_iter {
        if (usable_target_item_count + 1) > MAX_TARGET_ITEMS {
            break;
        }
        println!(
            "\n[Item {}/{} {:.4}%]",
            usable_target_item_count + 1,
            MAX_TARGET_ITEMS,
            (usable_target_item_count + 1) as f64 / MAX_TARGET_ITEMS as f64 * 100.0
        );

        let mut context_votes_per_user: HashMap<
            String,
            (Vec<Vote>, Vec<Vote>),
            rustc_hash::FxBuildHasher,
        > = FxHashMap::default();

        // for items in context, we only need the unique count
        let mut context_item_ids: HashSet<i64> = HashSet::new();

        {
            let unique_voters_on_target_item: Vec<User> = stmt_voters_on_note
                .query_map(params![target_item.id], |row| {
                    Ok(User {
                        id: row.get(0)?,
                        voted_at_millis: row.get(1)?,
                    })
                })?
                .filter_map(Result::ok)
                .dedup_by(|u1, u2| u1.id == u2.id)
                .collect();

            println!("target item id: {}", target_item.id);
            println!("text: {}", target_item.text);
            println!(
                "Number of unique users: {}",
                unique_voters_on_target_item.len()
            );

            // for every user who voted on the target item
            for user in unique_voters_on_target_item
                .iter()
                .take(MAX_USERS_VOTED_ON_TARGET_ITEM)
            {
                // votes on other items BEFORE voting on target item
                let votes_before = get_votes(
                    user.id.as_str(),
                    user.voted_at_millis,
                    &mut stmt_votes_before,
                )?;

                // votes on other items AFTER voting on target item
                let votes_after = get_votes(
                    user.id.as_str(),
                    user.voted_at_millis,
                    &mut stmt_votes_after,
                )?;
                // println!(
                //     "User {}, before: {}, after: {}",
                //     user.id,
                //     votes_before.len(),
                //     votes_after.len()
                // );

                // Only process this user if they have both before and after votes
                if votes_before.len() >= MIN_CONTEXT_VOTES_BEFORE_AND_AFTER
                    && votes_after.len() >= MIN_CONTEXT_VOTES_BEFORE_AND_AFTER
                {
                    for vote in votes_before.iter().chain(votes_after.iter()) {
                        context_item_ids.insert(vote.item_id);
                    }
                    context_votes_per_user.insert(user.id.clone(), (votes_before, votes_after));
                }
            }

            println!("Number of unique items: {}", context_item_ids.len());

            let (total_context_votes_before, total_context_votes_after) = context_votes_per_user
                .iter()
                .fold((0, 0), |(sum_before, sum_after), (_, (before, after))| {
                    (sum_before + before.len(), sum_after + after.len())
                });

            println!("total context votes before: {}", total_context_votes_before);
            println!("total context votes after:  {} ", total_context_votes_after);

            if context_item_ids.len() < 2
                || unique_voters_on_target_item.len() < MIN_USERS_VOTED_ON_TARGET_ITEM
                || total_context_votes_before == 0
                || total_context_votes_after == 0
            {
                println!("skipping");
                continue;
            }
            usable_target_item_count += 1;
        }

        if !target_item_idx.contains_key(&target_item.id) {
            target_item_idx.insert(target_item.id, target_item_idx.len());
        }

        let mut track_context_vote = |before: bool, vote: &Vote| {
            let virtual_user = VirtualUser {
                target_item_id: target_item.id,
                user_id: vote.user_id.clone(),
                before,
            };

            if !virtual_user_idx.contains_key(&virtual_user) {
                virtual_user_idx.insert(virtual_user.clone(), virtual_user_idx.len());
            }
            if !context_item_idx.contains_key(&vote.item_id) {
                context_item_idx.insert(vote.item_id, context_item_idx.len());
            }
            votes_by_virtual_users.insert((virtual_user.clone(), vote.item_id), vote.value);
            data.push(virtual_user, vote.item_id, vote.value as f32);
        };

        for (user_id, (context_votes_before, context_votes_after)) in context_votes_per_user.iter()
        {
            // println!(
            //     "User {}, before: {}, after: {}",
            //     user_id,
            //     context_votes_before.len(),
            //     context_votes_after.len()
            // );
            if !user_idx.contains_key(user_id) {
                user_idx.insert(user_id.clone(), user_idx.len());
            }
            for context_vote_before in context_votes_before.iter() {
                track_context_vote(true, context_vote_before);
            }
            for context_vote_after in context_votes_after.iter() {
                track_context_vote(false, context_vote_after);
            }
        }
    }

    println!("\nvirtual users: {}", virtual_user_idx.len());
    println!("total context items: {}", context_item_idx.len());
    println!("total unique users: {}", user_idx.len());
    println!("total target items: {}", target_item_idx.len());
    assert!(
        virtual_user_idx.len() % 2 == 0,
        "virtual users must always come in pairs"
    );

    let observed_context_matrix = build_observed_context_matrix(
        &votes_by_virtual_users,
        &context_item_idx,
        &virtual_user_idx,
    );

    // for (v, i) in virtual_user_idx.iter().sorted_by_key(|(_, &i)| i) {
    //     println!("{:>2}: {:?}", i, v);
    // }

    let top_context = 30.min(context_item_idx.len());
    let left_context = 80.min(virtual_user_idx.len());
    println!(
        "Observed context matrix: {:?}",
        observed_context_matrix.shape()
    );
    print_array(
        &observed_context_matrix
            .slice(s![..top_context, ..left_context])
            .to_owned(),
    );

    // println!("guess:");
    // print_array(
    //     &initial_guess(&observed_context_matrix)
    //         .slice(s![..top, ..left])
    //         .to_owned(),
    // );

    let recommender = RecommenderBuilder::new()
        .factors(CONTEXT_MATRIX_FACTORIZATION_RANK as u32)
        .iterations(1000)
        .fit_explicit(&data);

    // for user_id in recommender.user_ids() {
    //     let factors = recommender.user_factors(user_id);
    //     println!("{:?}: {:?}", user_id, factors);
    // }

    let completed_context_matrix = {
        let mut matrix: Array2<f32> =
            Array2::from_elem((context_item_idx.len(), virtual_user_idx.len()), f32::NAN);

        for virtual_user in recommender.user_ids() {
            for item_id in recommender.item_ids() {
                let context_item_index = *context_item_idx.get(item_id).unwrap();
                let virtual_user_index = *virtual_user_idx.get(virtual_user).unwrap();

                matrix[[context_item_index, virtual_user_index]] =
                    recommender.predict(virtual_user, item_id);
            }
        }

        matrix
    };

    // let factorization = matrix_factorization_svd(
    //     &observed_context_matrix,
    //     CONTEXT_MATRIX_FACTORIZATION_RANK,
    //     Some(initial_guess(&observed_context_matrix)),
    // );
    // let completed_context_matrix = factorization
    //     .u
    //     .dot(&Array2::from_diag(&factorization.s))
    //     .dot(&factorization.vt);
    //
    println!("Completed context matrix:");
    print_array32(
        &completed_context_matrix
            .slice(s![..top_context, ..left_context])
            .to_owned(),
    );
    println!("RMSE: {}", recommender.rmse(&data));

    // let virtual_user_embeddings = factorization.vt.t();

    // let change_matrix = calculate_change_matrix(
    //     virtual_user_embeddings.to_owned(),
    //     &virtual_user_idx,
    //     &target_item_idx,
    //     &user_idx,
    // );
    let (change_data, change_matrix) = {
        let mut matrix: Array2<f32> =
            Array2::from_elem((target_item_idx.len(), user_idx.len()), f32::NAN);
        let mut change_data = Dataset::new();

        for (virtual_user_before, virtual_user_after) in virtual_user_idx.keys().tuple_windows() {
            let embedding_before = Array1::from_vec(
                recommender
                    .user_factors(virtual_user_before)
                    .unwrap()
                    .to_vec(),
            );
            let embedding_after = Array1::from_vec(
                recommender
                    .user_factors(virtual_user_after)
                    .unwrap()
                    .to_vec(),
            );
            let change_vector = embedding_after.to_owned() - embedding_before.to_owned();
            let change_magnitude = change_vector.norm_l2();
            let target_item_index = target_item_idx[&virtual_user_before.target_item_id];
            let user_index = user_idx[&virtual_user_before.user_id];
            matrix[[target_item_index, user_index]] = change_magnitude;
            change_data.push(
                virtual_user_before.user_id.clone(),
                virtual_user_before.target_item_id,
                change_magnitude,
            );
        }
        (change_data, matrix)
    };

    let top_change_matrix = 30.min(target_item_idx.len());
    let left_change_matrix = 80.min(user_idx.len());
    println!("change matrix:");
    print_array32(
        &change_matrix
            .slice(s![..top_change_matrix, ..left_change_matrix])
            .to_owned(),
    );

    let change_recommender = RecommenderBuilder::new()
        .factors(CHANGE_MATRIX_FACTORIZATION_RANK as u32)
        .iterations(300)
        .fit_explicit(&change_data);

    let change_matrix_completion = {
        let mut matrix: Array2<f32> =
            Array2::from_elem((target_item_idx.len(), user_idx.len()), f32::NAN);

        for user_id in change_recommender.user_ids() {
            let user_index = user_idx[user_id];
            for target_item_id in change_recommender.item_ids() {
                let target_item_index = target_item_idx[target_item_id];

                matrix[[target_item_index, user_index]] =
                    change_recommender.predict(user_id, target_item_id);
            }
        }

        matrix
    };
    println!(
        "change matrix completion: {:?}",
        change_matrix_completion.shape()
    );
    print_array32(
        &(change_matrix_completion.clone() * 0.1)
            .slice(s![..top_change_matrix, ..left_change_matrix])
            .to_owned(),
    );

    let most_affected_user_by_target_item = find_most_affected_user_per_target_item(&change_matrix);
    println!("most affected users: {}", most_affected_user_by_target_item);

    let reverse_user_ids: FxHashMap<usize, String> =
        user_idx.iter().map(|(k, v)| (*v, k.clone())).collect();

    let reverse_target_item_ids: FxHashMap<usize, i64> =
        target_item_idx.iter().map(|(k, v)| (*v, *k)).collect();

    let reverse_context_item_ids: FxHashMap<usize, i64> =
        context_item_idx.iter().map(|(k, v)| (*v, *k)).collect();

    for (target_item_index, &user_index) in most_affected_user_by_target_item.iter().enumerate() {
        let target_item_id: i64 = reverse_target_item_ids[&target_item_index];
        let user_id: String = reverse_user_ids[&user_index].clone();
        let target_item_text: String = note_text(target_item_id, &mut stmt_note_text)?;
        let magnitude = change_matrix_completion[[target_item_index, user_index]];
        // TODO: track highest magnitude
        if magnitude < 1.8 {
            continue;
        }
        println!(
            "\n\ntarget_item: \"{}\" ({}) ",
            target_item_text, target_item_id,
        );
        println!("magnitude: {}, user: {}, ", magnitude, user_id);
        // find both virtual users: before and after voting on target item
        let virtual_user_before = VirtualUser {
            user_id: user_id.clone(),
            target_item_id,
            before: true,
        };
        let virtual_user_after = VirtualUser {
            user_id: user_id.clone(),
            target_item_id,
            before: false,
        };
        let predicted_changes = analyze_predicted_vote_changes(
            &completed_context_matrix,
            &virtual_user_before,
            &virtual_user_after,
            &virtual_user_idx,
            &reverse_context_item_ids,
            &mut stmt_note_text,
        )?;

        for pc in predicted_changes {
            println!(
                " |{:.2}| {:>5.2} -> {:>5.2} \"{}\"",
                pc.difference, pc.vote_before, pc.vote_after, pc.text
            );
        }

        // let context_item_ids_with_votes_before: Vec<(i64, f64)> = votes_by_virtual_users
        //     .iter()
        //     .filter_map(|((virtual_user, context_item_id), &value)| {
        //         if virtual_user == &virtual_user_before {
        //             Some((*context_item_id, value))
        //         } else {
        //             None
        //         }
        //     })
        //     .collect();
        //
        // for (item_id, vote) in context_item_ids_with_votes_before.iter() {
        //     let text: String =
        //         note_text(*item_id, &mut stmt_note_text).unwrap_or("<not found>".to_string());
        //     println!("before: {} on \"{}\"", vote, text);
        // }
        // let context_item_ids_with_votes_after: Vec<(i64, f64)> = votes_by_virtual_users
        //     .iter()
        //     .filter_map(|((virtual_user, context_item_id), &value)| {
        //         if virtual_user == &virtual_user_after {
        //             Some((*context_item_id, value))
        //         } else {
        //             None
        //         }
        //     })
        //     .collect();
        //
        // for (item_id, vote) in context_item_ids_with_votes_after.iter() {
        //     let text: String =
        //         note_text(*item_id, &mut stmt_note_text).unwrap_or("<not found>".to_string());
        //     println!("after: {} on \"{}\"", vote, text);
        // }
    }
    //
    // let change_matrix_factorization =
    //     matrix_factorization_svd(&change_matrix, CHANGE_MATRIX_FACTORIZATION_RANK, None);
    // let change_matrix_completion = change_matrix_factorization
    //     .u
    //     .dot(&Array2::from_diag(&change_matrix_factorization.s))
    //     .dot(&change_matrix_factorization.vt);
    //
    // println!("change matrix completion:");
    // print_array32(&change_matrix_completion));
    //
    // // TODO: print item texts which changed users minds on the target item
    // // Prepare the SQL statement
    // let _stmt = conn.prepare(
    //     "INSERT OR REPLACE INTO user_change (noteid, userid, change) VALUES (?1, ?2, ?3)",
    // )?;

    //     // TODO: save item with largest change
    //     let user_id = reverse_user_ids.get(&virtual_user_index).unwrap();
    //     stmt.execute(params![item_id, user_id, change_magnitude])?;
    // }
    //
    // let change_magnitude_avg = change_magnitude_sum / user_ids.len() as f64;
    //
    // println!("average mind change: {}", change_magnitude_avg);
    // conn.execute(
    //         "insert or replace into scores (noteId, change, users, items, before, after) values (?1, ?2, ?3, ?4, ?5, ?6)",
    //         (item_id, change_magnitude_avg, user_ids.len(), target_item_idx.len(), votes_before.len(), votes_after.len()),
    //     )?;

    Ok(())
}

fn build_observed_context_matrix(
    votes_by_virtual_users: &HashMap<(VirtualUser, i64), f64, rustc_hash::FxBuildHasher>,
    context_item_idx: &HashMap<i64, usize, rustc_hash::FxBuildHasher>,
    virtual_user_idx: &HashMap<VirtualUser, usize, rustc_hash::FxBuildHasher>,
) -> Array2<f64> {
    let mut matrix: Array2<f64> =
        Array2::from_elem((context_item_idx.len(), virtual_user_idx.len()), f64::NAN);

    for ((virtual_user, item_id), &vote_value) in votes_by_virtual_users.iter() {
        let context_item_index = *context_item_idx.get(item_id).unwrap();
        let virtual_user_index = *virtual_user_idx.get(virtual_user).unwrap();
        matrix[[context_item_index, virtual_user_index]] = vote_value;
    }

    matrix
}

fn calculate_change_matrix(
    virtual_user_embeddings: Array2<f64>,
    virtual_user_idx: &HashMap<VirtualUser, usize, rustc_hash::FxBuildHasher>,
    target_item_idx: &HashMap<i64, usize, rustc_hash::FxBuildHasher>,
    user_idx: &HashMap<String, usize, rustc_hash::FxBuildHasher>,
) -> Array2<f64> {
    let mut matrix: Array2<f64> =
        Array2::from_elem((target_item_idx.len(), user_idx.len()), f64::NAN);

    for ((virtual_user_before, &virtual_user_index_before), (_, &virtual_user_index_after)) in
        virtual_user_idx.iter().tuple_windows()
    {
        let embedding_before = virtual_user_embeddings.row(virtual_user_index_before);
        let embedding_after = virtual_user_embeddings.row(virtual_user_index_after);
        let change_vector = embedding_after.to_owned() - embedding_before.to_owned();
        let change_magnitude = change_vector.norm_l2();
        let target_item_index = target_item_idx[&virtual_user_before.target_item_id];
        let user_index = user_idx[&virtual_user_before.user_id];
        matrix[[target_item_index, user_index]] = change_magnitude;
    }
    matrix
}

fn find_most_affected_user_per_target_item(change_matrix: &Array2<f32>) -> Array1<usize> {
    change_matrix.map_axis(Axis(1), |row| {
        row.to_vec()
            .iter()
            .map(|&x| if x.is_nan() { -1.0 } else { x })
            .position_max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    })
}

fn analyze_predicted_vote_changes(
    completed_context_matrix: &Array2<f32>,
    virtual_user_before: &VirtualUser,
    virtual_user_after: &VirtualUser,
    virtual_user_idx: &HashMap<VirtualUser, usize, rustc_hash::FxBuildHasher>,
    reverse_context_item_ids: &HashMap<usize, i64, rustc_hash::FxBuildHasher>,
    stmt_note_text: &mut Statement,
) -> Result<Vec<PredictedVoteChange>, Box<dyn Error>> {
    let before_idx = virtual_user_idx[virtual_user_before];
    let after_idx = virtual_user_idx[virtual_user_after];

    let mut predicted_changes: Vec<PredictedVoteChange> = completed_context_matrix
        .rows()
        .into_iter()
        .enumerate()
        .map(|(context_item_index, row)| {
            let difference = (row[before_idx] - row[after_idx]).abs();
            let context_item_id = reverse_context_item_ids[&context_item_index];
            let text =
                note_text(context_item_id, stmt_note_text).unwrap_or("<no text>".to_string());
            PredictedVoteChange {
                vote_before: row[before_idx],
                vote_after: row[after_idx],
                difference,
                text,
            }
        })
        .collect();

    predicted_changes.sort_by(|a, b| b.difference.partial_cmp(&a.difference).unwrap());
    Ok(predicted_changes.into_iter().take(3).collect())
}

fn get_votes(
    user_id: &str,
    user_voted_at_millis: i64,
    statement: &mut Statement,
) -> Result<Vec<Vote>, Box<dyn Error>> {
    Ok(statement
        .query_map(
            params![user_id, user_voted_at_millis, MAX_CONTEXT_SIZE],
            |row| {
                Ok(Vote {
                    user_id: user_id.to_string(),
                    item_id: row.get(0)?,
                    value: if AGREE {
                        row.get(1)?
                    } else {
                        match row.get::<_, String>(1)?.as_str() {
                            "HELPFUL" => 1.0,
                            "SOMEWHAT_HELPFUL" => 0.0,
                            "NOT_HELPFUL" => -1.0,
                            _ => panic!(),
                        }
                    },
                })
            },
        )?
        .filter_map(Result::ok)
        .collect())
}

fn note_text(note_id: i64, stmt_note_text: &mut Statement) -> Result<String, Box<dyn Error>> {
    stmt_note_text
        .query_map(params![note_id], |row| row.get(0))?
        .next()
        .ok_or_else(|| Box::<dyn Error>::from(format!("note not found: {}", note_id)))?
        .map_err(Box::<dyn Error>::from)
}

#[derive(Debug)]
struct User {
    id: String,
    voted_at_millis: i64,
}

#[derive(Debug)]
struct Item {
    id: i64,
    text: String,
}

#[derive(Debug, Clone)]
struct Vote {
    user_id: String,
    item_id: i64,
    value: f64,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct VirtualUser {
    target_item_id: i64,
    user_id: String,
    before: bool,
}

#[derive(Debug)]
struct PredictedVoteChange {
    vote_before: f32,
    vote_after: f32,
    difference: f32,
    text: String,
}
