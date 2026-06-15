import pandas as pd
import streamlit as st


st.title("Member Classification Model")


# upload files
members_file = st.file_uploader("Upload Members Information File",
                                type=["csv", "xlsx"])
attendance_file = st.file_uploader("Upload Attendance File",
                                   type=["csv", "xlsx"])
payment_file = st.file_uploader("Upload Payment File",
                                type=["csv", "xlsx"])

if members_file and attendance_file and payment_file:
    # read files
    members_df = pd.read_csv(members_file) if members_file.name.endswith(
        '.csv') else pd.read_excel(members_file)
    attendance_df = pd.read_csv(attendance_file) if attendance_file.name.endswith(
        '.csv') else pd.read_excel(attendance_file)
    payment_df = pd.read_csv(payment_file) if payment_file.name.endswith(
        '.csv') else pd.read_excel(payment_file)

    # clean and process data for classification model
    for df in [members_df, attendance_df, payment_df]:
        df.columns = df.columns.str.strip()

    # validation
    required_members = ["MemberID", "RegDate"]
    required_attendance = ["MemberID", "Date"]
    required_payments = ["MemberID", "Amount", "PaymentDate"]

    def validate(df, cols, name):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            st.error(f"{name} missing columns: {', '.join(missing)}")
            return False
        return True
    
    if (
        validate(members_df, required_members, "Members")
        and validate(attendance_df, required_attendance, "Attendance")
        and validate(payment_df, required_payments, "Payments")
    ):
        

        # convert date columns to datetime
        members_df["RegDate"] = pd.to_datetime(members_df["RegDate"], errors="coerce")
        attendance_df["Date"] = pd.to_datetime(attendance_df["Date"], errors="coerce")


        payment_df["Amount"] = pd.to_numeric(payment_df["Amount"], errors="coerce").fillna(0)
        payment_df["PaymentDate"] = pd.to_datetime(payment_df["PaymentDate"], errors="coerce")
        current_year = pd.Timestamp.today().year

        attendance_summary = (
            attendance_df.groupby("MemberID", sort=False)
            .size()
            .reset_index(name="Meetings")
        )

        payment_df["PaymentYear"] = payment_df["PaymentDate"].dt.year
        three_year_window = payment_df["PaymentYear"].between(current_year - 2, current_year)

        yearly_paid = (
            payment_df.loc[three_year_window]
            .groupby(["MemberID", "PaymentYear"])["Amount"]
            .sum()
            .reset_index(name="YearlyPaid")
        )

        yearly_paid["PaidInYear"] = yearly_paid["YearlyPaid"] > 0

        payment_summary = (
            yearly_paid.groupby("MemberID")
            .agg(
                TotalPaid=("YearlyPaid", "sum"),
                ConsecutiveYears=("PaymentYear", "nunique"),
                PaidYears=("PaidInYear", "sum")
            )
            .reset_index()
        )

        payment_summary["PaidThreeYears"] = (
            (payment_summary["ConsecutiveYears"] == 3) &
            (payment_summary["PaidYears"] == 3)
        )

        result = members_df.merge(attendance_summary, on="MemberID", how="left") \
            .merge(payment_summary, on="MemberID", how="left")
        
        # FILL IN THE MISSING VALUES AFTER MERGING THE DATAFRAMES
        result["Meetings"] = result["Meetings"].fillna(0).astype(int)
        result["TotalPaid"] = result["TotalPaid"].fillna(0)
        result["PaidThreeYears"] = result["PaidThreeYears"].fillna(False)

        def classify_members(df, payment_threshold=120, attendance_threshold=1):
            processed_df = df.copy()
            processed_df["RegDate"] = processed_df["RegDate"].fillna(pd.Timestamp("2023-01-01"))

            condition = (
                (processed_df["RegDate"] <= pd.Timestamp.today()) &
                (processed_df["PaidThreeYears"] == True) &
                (processed_df["Meetings"] >= attendance_threshold)
            )

            processed_df["Status"] = condition.map({True: "Active", False: "Inactive"})
            return processed_df

        classified_df = classify_members(result)

        st.success("Classification completed successfully.")
        st.dataframe(classified_df[["MemberID", "RegDate", "Meetings", "TotalPaid", "PaidThreeYears", "Status"]])