import pandas as pd
import streamlit as st
import openpyxl

st.set_page_config(
    page_title="Member Classification Model",
    page_icon="📋",
    layout="wide"
)

st.title("Member Classification Model")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def load_file(uploaded_file):
    """Load CSV or Excel file."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def validate_columns(df, required_columns, file_name):
    """Validate required columns."""
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        st.error(
            f"{file_name} is missing required column(s): "
            + ", ".join(missing)
        )
        return False

    return True


def clean_member_data(df):
    df = df.copy()

    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["MemberID"])

    duplicates = df[df["MemberID"].duplicated()]

    if not duplicates.empty:
        st.warning(
            f"Members file contains {len(duplicates)} duplicate MemberID values."
        )

    df = df.drop_duplicates(subset=["MemberID"], keep="first")

    df["RegDate"] = pd.to_datetime(
        df["RegDate"],
        errors="coerce"
    )

    invalid = df["RegDate"].isna().sum()

    if invalid > 0:
        st.warning(
            f"{invalid} invalid registration dates were replaced with "
            "2022-01-01."
        )

    df["RegDate"] = df["RegDate"].fillna(
        pd.Timestamp("2022-01-01")
    )

    return df


def clean_attendance(df):
    df = df.copy()

    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["MemberID"])

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Year"] = df["Date"].dt.year

    before = len(df)

    df = df.dropna(subset=["Year"])

    removed = before - len(df)

    if removed > 0:
        st.warning(
            f"Removed {removed} attendance records with invalid dates."
        )

    df["Year"] = df["Year"].astype(int)

    return df


def clean_payments(df):
    df = df.copy()

    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["MemberID"])

    df["PaymentDate"] = pd.to_datetime(
        df["PaymentDate"],
        errors="coerce"
    )

    df["PaymentYear"] = df["PaymentDate"].dt.year

    before = len(df)

    df = df.dropna(subset=["PaymentYear"])

    removed = before - len(df)

    if removed > 0:
        st.warning(
            f"Removed {removed} payment records with invalid dates."
        )

    df["PaymentYear"] = df["PaymentYear"].astype(int)

    df["Amount"] = pd.to_numeric(
        df["Amount"],
        errors="coerce"
    ).fillna(0)

    return df


# ---------------------------------------------------------
# Upload Files
# ---------------------------------------------------------

members_file = st.file_uploader(
    "Upload Members File",
    type=["csv", "xlsx"]
)

attendance_file = st.file_uploader(
    "Upload Attendance File",
    type=["csv", "xlsx"]
)

payment_file = st.file_uploader(
    "Upload Payments File",
    type=["csv", "xlsx"]
)

# ---------------------------------------------------------
# Process
# ---------------------------------------------------------

if members_file and attendance_file and payment_file:

    members_df = load_file(members_file)
    attendance_df = load_file(attendance_file)
    payment_df = load_file(payment_file)

    required_members = [
        "MemberID",
        "RegDate"
    ]

    required_attendance = [
        "MemberID",
        "Date"
    ]

    required_payments = [
        "MemberID",
        "PaymentDate",
        "Amount"
    ]

    valid = (
        validate_columns(
            members_df,
            required_members,
            "Members File"
        )
        and
        validate_columns(
            attendance_df,
            required_attendance,
            "Attendance File"
        )
        and
        validate_columns(
            payment_df,
            required_payments,
            "Payments File"
        )
    )

    if valid:

        members_df = clean_member_data(members_df)
        attendance_df = clean_attendance(attendance_df)
        payment_df = clean_payments(payment_df)

        # Safe evaluation year

        if attendance_df.empty:
            max_year = pd.Timestamp.today().year
        else:
            max_year = int(attendance_df["Year"].max())

        target_year = st.sidebar.selectbox(
            "Evaluation Year",
            options=range(max_year - 2, max_year + 2),
            index=2
        )

        required_years = [
            target_year - 2,
            target_year - 1,
            target_year
        ]

        # ---------------------------------------------------------
        # Attendance Rule
        # Veterans must attend at least one meeting in each of the
        # last three evaluation years.
        # ---------------------------------------------------------

        attendance_filtered = attendance_df[
            attendance_df["Year"].isin(required_years)
        ]

        attendance_yearly = (
            attendance_filtered
            .groupby(["MemberID", "Year"])
            .size()
            .reset_index(name="Meetings")
        )

        if attendance_yearly.empty:

            attendance_pivot = pd.DataFrame(
                index=members_df["MemberID"].unique()
            )

        else:

            attendance_pivot = attendance_yearly.pivot_table(
                index="MemberID",
                columns="Year",
                values="Meetings",
                fill_value=0
            )

        for year in required_years:
            if year not in attendance_pivot.columns:
                attendance_pivot[year] = 0

        attendance_pivot = attendance_pivot.reindex(
            columns=required_years,
            fill_value=0
        )

        attendance_pivot["MeetsAttendanceRule"] = (
            attendance_pivot[required_years] >= 1
        ).all(axis=1)

        attendance_summary = (
            attendance_pivot[["MeetsAttendanceRule"]]
            .reset_index()
        )

        # ---------------------------------------------------------
        # Payment Rule
        # Payments are evaluated PER YEAR, not as one total.
        # ---------------------------------------------------------

        payment_yearly = (
            payment_df
            .groupby(["MemberID", "PaymentYear"])["Amount"]
            .sum()
            .reset_index()
        )

        payment_pivot = payment_yearly.pivot_table(
            index="MemberID",
            columns="PaymentYear",
            values="Amount",
            fill_value=0
        )

        result = members_df.merge(
            attendance_summary,
            on="MemberID",
            how="left"
        )

        result["MeetsAttendanceRule"] = (
            result["MeetsAttendanceRule"]
            .fillna(False)
        )

        evaluation_date = pd.Timestamp(
            f"{target_year}-12-31"
        )

        result["YearsRegistered"] = (
            (
                evaluation_date
                - result["RegDate"]
            ).dt.days
            / 365.25
        )

        veteran_status = []
        newcomer_status = []

        for _, row in result.iterrows():

            member = row["MemberID"]

            registration_year = row["RegDate"].year

            years_registered = row["YearsRegistered"]

            # ----------------------------------------
            # Veteran (>2 years)
            # ----------------------------------------

            if years_registered >= 2:

                attendance_ok = bool(
                    row["MeetsAttendanceRule"]
                )

                payment_ok = True

                for year in required_years:

                    amount = 0

                    if (
                        member in payment_pivot.index
                        and year in payment_pivot.columns
                    ):
                        amount = payment_pivot.loc[
                            member,
                            year
                        ]

                    if amount < 120:
                        payment_ok = False
                        break

                veteran_status.append(
                    attendance_ok and payment_ok
                )

                newcomer_status.append(False)

            # ----------------------------------------
            # New Member (less than 2 years)
            # Must pay at least $120 for every year
            # since registration.
            # No attendance requirement.
            # ----------------------------------------

            else:

                payment_ok = True

                years_to_check = list(
                    range(
                        registration_year,
                        target_year + 1
                    )
                )

                for year in years_to_check:

                    amount = 0

                    if (
                        member in payment_pivot.index
                        and year in payment_pivot.columns
                    ):
                        amount = payment_pivot.loc[
                            member,
                            year
                        ]

                    if amount < 120:
                        payment_ok = False
                        break

                newcomer_status.append(payment_ok)

                veteran_status.append(False)

        result["VeteranActive"] = veteran_status
        result["IsNewcomerActive"] = newcomer_status

        # ---------------------------------------------------------
        # Final Classification
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Final Classification
        # ---------------------------------------------------------

        classified_df = result.copy()

        classified_df["Status"] = "Inactive"

        classified_df.loc[
            classified_df["VeteranActive"] |
            classified_df["IsNewcomerActive"],
            "Status"
        ] = "Active"


        # ---------------------------------------------------------
        # Sponsorship Eligibility
        # Active for 3 consecutive evaluation years
        # ---------------------------------------------------------

        years_to_check = [
            target_year - 2,
            target_year - 1,
            target_year
        ]

        active_history = {}

        for year in years_to_check:

            # Attendance for this year
            yearly_attendance = (
                attendance_df[
                    attendance_df["Year"] == year
                ]
                .groupby("MemberID")
                .size()
            )


            # Payment for this year
            yearly_payment = (
                payment_df[
                    payment_df["PaymentYear"] == year
                ]
                .groupby("MemberID")["Amount"]
                .sum()
            )


            yearly_status = pd.DataFrame({
                "Attendance": yearly_attendance,
                "Payment": yearly_payment
            }).fillna(0)


            yearly_status["Active"] = (
                (yearly_status["Attendance"] >= 1)
                &
                (yearly_status["Payment"] >= 120)
            )


            active_history[year] = (
                yearly_status["Active"]
            )


        sponsorship_history = pd.DataFrame(
            active_history
        ).fillna(False)


        sponsorship_history["EligibleForSponsorship"] = (
            sponsorship_history[
                years_to_check
            ]
            .all(axis=1)
        )


        sponsorship_history = (
            sponsorship_history[
                ["EligibleForSponsorship"]
            ]
            .reset_index()
        )


        classified_df = classified_df.merge(
            sponsorship_history,
            on="MemberID",
            how="left"
        )


        classified_df["EligibleForSponsorship"] = (
            classified_df["EligibleForSponsorship"]
            .fillna(False)
        )


        # ---------------------------------------------------------
        # Display Metrics
        # ---------------------------------------------------------

        st.success(
            "Classification completed successfully!"
        )


        total_members = len(classified_df)

        active_members = (
            classified_df["Status"] == "Active"
        ).sum()

        inactive_members = (
            classified_df["Status"] == "Inactive"
        ).sum()

        sponsorship_members = (
            classified_df["EligibleForSponsorship"]
            .sum()
        )


        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "Total Members",
            total_members
        )


        col2.metric(
            "Active Members",
            f"{active_members} "
            f"({active_members/total_members:.1%})"
            if total_members else "0"
        )


        col3.metric(
            "Inactive Members",
            f"{inactive_members} "
            f"({inactive_members/total_members:.1%})"
            if total_members else "0"
        )


        col4.metric(
            "Sponsorship Eligible",
            sponsorship_members
        )


        # ---------------------------------------------------------
        # Display Results
        # ---------------------------------------------------------

        display_columns = [
            "MemberID",
            "RegDate",
            "YearsRegistered",
            "VeteranActive",
            "IsNewcomerActive",
            "Status",
            "EligibleForSponsorship"
        ]


        st.subheader(
            "Classification Results"
        )


        st.dataframe(
            classified_df[
                display_columns
            ].sort_values("MemberID"),
            use_container_width=True
        )


        # ---------------------------------------------------------
        # Download Results
        # ---------------------------------------------------------

        csv = (
            classified_df
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            label="📥 Download Classification CSV",
            data=csv,
            file_name=f"classified_members_{target_year}.csv",
            mime="text/csv"
        )


        # ---------------------------------------------------------
        # Summary
        # ---------------------------------------------------------

        st.subheader(
            "Summary"
        )


        summary = (
            classified_df["Status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Count")
        )


        summary["Percentage"] = (
            summary["Count"]
            / total_members
            * 100
        ).round(1)


        st.dataframe(
            summary,
            use_container_width=True
        )