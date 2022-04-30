library(baseballr)
library(tfestimators)
library(dplyr)
library(tensorflow)
library(modelr)

res_classify <- function(des, la, ls) {
    ifelse((des == "ball" | des == "called_strike" | des == "blocked_ball"),
        "Take", "Swing")
    # ifelse((des == "swinging_strike" | des == "swinging_strike_blocked" | des == "foul_tip"), # nolint
    #     "Weak/Miss",
    # ifelse((la <= -10 | la >= 50 | ls <= 75),
    #     "Weak/Miss",
    # "Strong")))
}

get_pbp_year <- function(year) {
    start <- paste(year, "01-01", sep = "-")
    end <- paste(year, "12-31", sep = "-")
    pid <- playerid_lookup(last_name = "Devers", first_name = "Rafael") %>%
    select(mlbam_id)
    df <- scrape_statcast_savant_batter(start_date = start,
                                        end_date = end,
                                        batterid = strtoi(pid)) %>%
    filter(release_spin_rate != "NA") %>%
    select(release_speed,
            balls,
            strikes,
            pfx_x,
            pfx_z,
            plate_x,
            plate_z,
            release_spin_rate,
            description,
            launch_angle,
            launch_speed
            ) %>%
    mutate(class = res_classify(description, launch_angle, launch_speed)) %>%
    filter(class != "NA")
    return(df)
}

df_total <- data.frame()
for (y in 2019:2021) {
    df_total <- rbind(df_total, get_pbp_year(y))
}
df_total <- df_total[-c(9:11)]

write.csv(df_total, "/Users/alexchandy13/Documents/School/6 Spring 2022/CSE 5713/Project/Code/bin-df.csv", row.names = FALSE) # nolint
