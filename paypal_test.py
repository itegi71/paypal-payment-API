import  paypalrestsdk
import webbrowser 

paypalrestsdk.configure({
    "mode":"sandbox",
    "client_id":"AW2FR-vpnmY5LE64hgfHPz4BxdaCcb0rdt61AHkpMAQAIewAvZKNWfmox407C3YXHxaccx_Js4DwKVR3",
    "client_secret":"EIuhSos_xBiNpgIkD5T3wkvLdwTrQ-WYi9JAX66LnTqbpFYTKPD00QyonMHYB3tgy0aU_zpcaZ50HB6v"

})

#create a payment
print("Creating a test payment for $1.00...")

#info we  send to paypal

payment_info={
    "intent":"sale",
    "payer":{
        "payment_method":"paypal"

    },
    "redirect_urls":{
        "return_url":"http://example.com/success",
        "cancel_url":"http://example.com/cancel"

    },

    "transactions":[{
        "amount":{
            "total":"1.00",
            "currency":"USD"
        },
        "description":"this is a test payment for my tutorial"

    }]
}

#create the payment on paypals servers

payment =paypalrestsdk.Payment(payment_info)

#try
if payment.create():
    print("Payment created successfully!")

    #link where user can approve the paymennt 

    for link in payment.links :
        if link.rel=="approval_url":
            approval_url=link.href

            print("now opening the paypal website for you to approve the payment...")

            webbrowser.open(approval_url)
            break
else:
    print("oh no  ! there was an error creating the payment:")
    print(payment.error)